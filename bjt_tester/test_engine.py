"""
test_engine.py - 自动化测试引擎
实现双嵌套扫描状态机和五项 BJT 参数测试函数。
通过 BaseInstrumentDriver 接口调用驱动，与具体实现解耦。
"""
import os
import pathlib
import time as _time
import threading
import numpy as np
from typing import List, Optional, Callable
from driver_base import BaseInstrumentDriver
from data_processor import (trimmed_mean, remove_outliers,
                            calculate_hfe, calculate_leakage,
                            process_adc_samples)

# ===== 原始数据保存开关 =====
SAVE_RAW_DATA = True
RAW_DATA_BASE = str(pathlib.Path(__file__).parent.parent / "raw_data")

# ===== 测试常量 =====
R_BASE = 1e3            # 基极限流电阻 (Ohm) — hFE 测试用（硬件已改为 1kΩ）
R_BASE_SAT = 1e3        # 基极限流电阻 (Ohm) — VCE(sat) 测试用
R_SENSE = 100.0         # 集电极采样电阻 (Ohm)
R_LEAK = 10e6           # 跨阻放大器反馈电阻 (Ohm)
R_LIMIT = 1e6           # BVCEO 限流电阻 (Ohm)
BUFFER_SIZE = 2048      # ADC 采样缓冲区
TRIM_RATIO = 0.05       # 截尾比例 5%

# ===== 通道定义 =====
CH_VB = 0               # AWG CH0: 基极驱动
CH_VCE = 1              # AWG CH1: 集电极偏置
CH_VBE = 0              # ADC CH0: VBE 采集
CH_VCE_SENSE = 1        # ADC CH1: VCE 采集

# ===== 安全阈值 =====
IC_MAX = 50e-3          # 最大集电极电流 50 mA
VCE_MAX_REAL = 5.0      # 真实硬件最大电压 5 V (AWG 上限)
VCE_MAX_VIRTUAL = 20.0  # 虚拟仿真允许的最大电压
LEAKAGE_TREND_MAX_V = 5.0   # 漏电流趋势测试最大电压
LEAKAGE_TREND_STEP = 0.5    # 漏电流趋势测试步进
LEAKAGE_5V_THRESHOLD = 1e-6 # 5V 时漏电流判定阈值 1 μA

# ===== 判定标准 (兼容 2N3904 / BC337 全系列) =====
HFE_MIN, HFE_MAX = 100, 630
VCESAT_MAX = 0.4        # V
VBESAT_MAX = 1.0        # V
ICBO_MAX = 100e-9       # A
ICEO_MAX = 200e-9       # A
BVCEO_MIN = 40.0        # V


class HardwareError(RuntimeError):
    """硬件异常：接线错误、器件未插入等可诊断的硬件问题"""
    def __init__(self, message: str, suggestion: str = ""):
        super().__init__(message)
        self.suggestion = suggestion


class TestEngine:
    """
    BJT 全参数自动化测试引擎。
    通过构造函数注入驱动实例，实现与硬件的解耦。
    """

    def __init__(self, driver: BaseInstrumentDriver):
        self.driver = driver
        self._progress_callback: Optional[Callable] = None
        self._log_callback: Optional[Callable] = None
        self._user_prompt_callback: Optional[Callable[[str], None]] = None
        self._stop_event = threading.Event()

    def set_callbacks(self, progress_cb=None, log_cb=None, user_prompt_cb=None):
        """设置进度、日志和用户提示回调函数"""
        self._progress_callback = progress_cb
        self._log_callback = log_cb
        self._user_prompt_callback = user_prompt_cb

    def request_stop(self):
        """外部请求停止测试"""
        self._stop_event.set()

    def _check_stop(self):
        """检查是否收到停止请求，若是则执行紧急停止并抛出异常"""
        if self._stop_event.is_set():
            self.driver.emergency_stop()
            raise InterruptedError("用户触发紧急停止")

    def _log(self, msg: str):
        if self._log_callback:
            self._log_callback(msg)

    def _progress(self, pct: int, text: str):
        if self._progress_callback:
            self._progress_callback(pct, text)

    def _prompt_user(self, msg: str):
        """阻塞等待用户确认（如改接线提示）。虚拟模式跳过。"""
        if self._is_virtual():
            return
        if self._user_prompt_callback:
            self._user_prompt_callback(msg)
        else:
            self._log(f"[需要操作] {msg}")

    def _is_virtual(self):
        return hasattr(self.driver, 'simulate_hfe_measurement')

    def set_data_context(self, device_name: str, test_round: int):
        """设置原始数据保存的上下文信息（由 GUI 在每次测试前调用）。"""
        self._device_name = device_name
        self._test_round = test_round

    def _get_data_dir(self) -> str:
        """获取当前测试上下文的原始数据保存目录并确保存在。"""
        if not SAVE_RAW_DATA:
            return ""
        dn = getattr(self, '_device_name', 'unknown')
        tr = getattr(self, '_test_round', 0)
        d = os.path.join(RAW_DATA_BASE, dn, f"test{tr}")
        os.makedirs(d, exist_ok=True)
        return d

    def _save_raw(self, filename: str, data):
        """保存原始 ADC 数据为 .npy 文件。"""
        if not SAVE_RAW_DATA:
            return
        d = self._get_data_dir()
        if d:
            np.save(os.path.join(d, filename), np.array(data, dtype=np.float64))

    def _safety_check(self, ic: float, vce: float):
        """安全检查：过流/过压保护"""
        if abs(ic) > IC_MAX:
            self.driver.emergency_stop()
            raise RuntimeError(
                f"过流保护: IC={ic*1e3:.1f}mA > {IC_MAX*1e3:.0f}mA")
        vce_limit = VCE_MAX_VIRTUAL if self._is_virtual() else VCE_MAX_REAL
        if abs(vce) > vce_limit:
            self.driver.emergency_stop()
            raise RuntimeError(
                f"过压保护: VCE={vce:.1f}V > {vce_limit:.0f}V")

    def _sanity_check_hfe(self, params: dict, vbe: float):
        """hFE 测量后的合理性检查（仅 REAL 模式）"""
        if self._is_virtual():
            return
        ic = params["ic"]
        hfe = params["hfe"]
        if vbe < 0.2:
            raise HardwareError(
                f"VBE={vbe:.3f}V，过低，BJT 可能未正确插入",
                "检查BJT是否正确插入测试座，确认E/B/C引脚对应关系")
        if vbe > 1.0:
            raise HardwareError(
                f"VBE={vbe:.3f}V，过高，基极电路可能开路",
                "检查基极限流电阻(1kΩ)连接是否正常")
        if ic < 1e-9:
            raise HardwareError(
                "集电极电流接近零，BJT 可能未正确插入或未导通",
                "检查BJT是否正确插入测试座，确认E/B/C引脚对应关系")
        if hfe < 1.0:
            raise HardwareError(
                f"hFE={hfe:.2f}，远低于正常范围，BJT 可能反接或损坏",
                "检查BJT的E/C引脚是否接反，或更换器件重新测试")
        if hfe > 10000:
            raise HardwareError(
                f"hFE={hfe:.0f}，异常偏高，可能存在短路",
                "检查测试电路是否有短路，特别是集电极和基极之间")

    def _sanity_check_vcesat(self, vce_sat: float):
        """VCE(sat) 测量后的合理性检查（仅 REAL 模式）"""
        if self._is_virtual():
            return
        if vce_sat < 0:
            raise HardwareError(
                f"VCE(sat)={vce_sat:.3f}V 为负值，Sense 线可能接反",
                "检查 Kelvin 四线制 Sense 线的正负极性")
        if vce_sat > 2.0:
            raise HardwareError(
                f"VCE(sat)={vce_sat:.3f}V 异常偏高，BJT 可能未饱和",
                "检查基极驱动电流是否足够，或BJT是否损坏")

    def test_hFE(self, ib_list: List[float] = None,
                 vce_set: float = 5.0) -> List[dict]:
        """
        直流电流增益测试。
        固定 VCE，扫描 IB，测量 IC，计算 hFE = IC / IB。
        """
        if ib_list is None:
            ib_list = [50e-6, 100e-6, 150e-6, 200e-6]
        results = []
        self.driver.set_voltage(CH_VCE, vce_set)

        for i_ib, ib_target in enumerate(ib_list):
            vb_set = ib_target * R_BASE + 0.7
            self.driver.set_voltage(CH_VB, vb_set)

            if not hasattr(self.driver, 'simulate_hfe_measurement'):
                _time.sleep(0.05)

            # 判断驱动类型，使用对应的采集方式
            if hasattr(self.driver, 'simulate_hfe_measurement'):
                sim = self.driver.simulate_hfe_measurement(
                    vb_set, vce_set, BUFFER_SIZE)
                raw_vbe = sim["vbe_data"]
                raw_vce = sim["vce_data"]
            else:
                if hasattr(self.driver, 'read_dual_voltage'):
                    raw_vbe, raw_vce = self.driver.read_dual_voltage(
                        CH_VBE, CH_VCE_SENSE, BUFFER_SIZE)
                else:
                    raw_vbe = self.driver.read_voltage(CH_VBE, BUFFER_SIZE)
                    raw_vce = self.driver.read_voltage(CH_VCE_SENSE, BUFFER_SIZE)

            # 保存原始 ADC 数据
            ib_ua = int(ib_target * 1e6)
            self._save_raw(f"hfe_ch1_vbe_ib{ib_ua}uA.npy", raw_vbe)
            self._save_raw(f"hfe_ch2_vce_ib{ib_ua}uA.npy", raw_vce)

            vbe = trimmed_mean(raw_vbe, TRIM_RATIO, use_sigma=True)
            vce_m = trimmed_mean(raw_vce, TRIM_RATIO, use_sigma=True)
            params = calculate_hfe(vb_set, vbe, vce_set, vce_m,
                                   R_BASE, R_SENSE)
            self._safety_check(params["ic"], vce_m)
            if len(results) == 0:
                self._sanity_check_hfe(params, vbe)
            results.append(params)

        self.driver.set_voltage(CH_VB, 0)
        self.driver.set_voltage(CH_VCE, 0)
        return results

    def test_vce_sat(self, ic_set: float = 10e-3,
                     ib_set: float = 1e-3) -> float:
        """饱和压降测试。Kelvin 四线制差分测量 VCE(sat)。"""

        if hasattr(self.driver, 'simulate_vcesat_measurement'):
            sim = self.driver.simulate_vcesat_measurement(
                ic_set, ib_set, BUFFER_SIZE)
            self._save_raw("vcesat_sense_p.npy", sim["sense_p"])
            self._save_raw("vcesat_sense_n.npy", sim["sense_n"])
            vce_sat = trimmed_mean(sim["sense_p"], TRIM_RATIO, use_sigma=True) - \
                      trimmed_mean(sim["sense_n"], TRIM_RATIO, use_sigma=True)
        else:
            self.driver.set_voltage(CH_VB, 0)
            self.driver.set_voltage(CH_VCE, 0)
            _time.sleep(0.05)

            vb = ib_set * R_BASE_SAT + 0.7
            self.driver.set_voltage(CH_VB, vb)
            vce_force = ic_set * R_SENSE + 0.3
            self.driver.set_voltage(CH_VCE, vce_force)
            _time.sleep(0.2)

            if hasattr(self.driver, 'read_dual_voltage'):
                raw_sp, raw_sn = self.driver.read_dual_voltage(
                    0, 1, BUFFER_SIZE)
            else:
                raw_sn = self.driver.read_voltage(CH_VCE_SENSE, BUFFER_SIZE)
                raw_sp = self.driver.read_voltage(CH_VBE, BUFFER_SIZE)

            self._save_raw("vcesat_sense_p.npy", raw_sp)
            self._save_raw("vcesat_sense_n.npy", raw_sn)
            vce_sat = trimmed_mean(raw_sn, TRIM_RATIO, use_sigma=True) - \
                      trimmed_mean(raw_sp, TRIM_RATIO, use_sigma=True)

        self._sanity_check_vcesat(vce_sat)
        self.driver.set_voltage(CH_VB, 0)
        self.driver.set_voltage(CH_VCE, 0)
        return vce_sat

    def test_vbe_sat(self, ic_set: float = 10e-3,
                     ib_set: float = 1e-3) -> float:
        """基极饱和压降测试。Kelvin 四线制差分测量 VBE(sat)。
        需要 ADC CH1 接基极、CH0 接发射极。
        """

        if hasattr(self.driver, 'simulate_vbesat_measurement'):
            sim = self.driver.simulate_vbesat_measurement(
                ic_set, ib_set, BUFFER_SIZE)
            self._save_raw("vbesat_sense_p.npy", sim["sense_p"])
            self._save_raw("vbesat_sense_n.npy", sim["sense_n"])
            vbe_sat = trimmed_mean(sim["sense_p"], TRIM_RATIO, use_sigma=True) - \
                      trimmed_mean(sim["sense_n"], TRIM_RATIO, use_sigma=True)
        else:
            self.driver.set_voltage(CH_VB, 0)
            self.driver.set_voltage(CH_VCE, 0)
            _time.sleep(0.05)

            vb = ib_set * R_BASE_SAT + 0.7
            self.driver.set_voltage(CH_VB, vb)
            vce_force = ic_set * R_SENSE + 0.3
            self.driver.set_voltage(CH_VCE, vce_force)
            _time.sleep(0.2)

            if hasattr(self.driver, 'read_dual_voltage'):
                raw_sp, raw_sn = self.driver.read_dual_voltage(
                    0, 1, BUFFER_SIZE)
            else:
                raw_sp = self.driver.read_voltage(CH_VBE, BUFFER_SIZE)
                raw_sn = self.driver.read_voltage(CH_VCE_SENSE, BUFFER_SIZE)

            self._save_raw("vbesat_sense_p.npy", raw_sp)
            self._save_raw("vbesat_sense_n.npy", raw_sn)
            vbe_sat = trimmed_mean(raw_sn, TRIM_RATIO, use_sigma=True) - \
                      trimmed_mean(raw_sp, TRIM_RATIO, use_sigma=True)

        self.driver.set_voltage(CH_VB, 0)
        self.driver.set_voltage(CH_VCE, 0)
        return vbe_sat

    def test_icbo(self, vcb_virtual: float = 20.0,
                 vcb_real: float = 5.0) -> float:
        """
        集电极-基极反向截止电流测试。
        虚拟仿真使用 20V 匹配 datasheet 条件，真实硬件使用 5V。
        """
        if hasattr(self.driver, 'simulate_icbo_measurement'):
            sim = self.driver.simulate_icbo_measurement(
                vcb_virtual, BUFFER_SIZE)
            raw = sim["transimpedance_data"]
            self._save_raw("icbo_transimpedance.npy", raw)
            v_out = trimmed_mean(raw, TRIM_RATIO, use_sigma=True)
        else:
            self.driver.set_voltage(CH_VCE, vcb_real)
            self.driver.set_voltage(CH_VB, 0)
            raw = self.driver.read_voltage(0, BUFFER_SIZE)
            self._save_raw("icbo_transimpedance.npy", raw)
            v_out = trimmed_mean(raw, TRIM_RATIO, use_sigma=True)

        icbo = calculate_leakage(v_out, R_LEAK)
        self.driver.set_voltage(CH_VCE, 0)
        return icbo

    def test_iceo(self, vce_virtual: float = 20.0,
                  vce_real: float = 5.0) -> float:
        """
        集电极-发射极反向截止电流测试。
        虚拟仿真使用 20V 匹配 datasheet 条件，真实硬件使用 5V。
        """
        if hasattr(self.driver, 'simulate_iceo_measurement'):
            sim = self.driver.simulate_iceo_measurement(
                vce_virtual, BUFFER_SIZE)
            raw = sim["transimpedance_data"]
            self._save_raw("iceo_transimpedance.npy", raw)
            v_out = trimmed_mean(raw, TRIM_RATIO, use_sigma=True)
        else:
            self.driver.set_voltage(CH_VCE, vce_real)
            raw = self.driver.read_voltage(0, BUFFER_SIZE)
            self._save_raw("iceo_transimpedance.npy", raw)
            v_out = trimmed_mean(raw, TRIM_RATIO, use_sigma=True)

        iceo = calculate_leakage(v_out, R_LEAK)
        self.driver.set_voltage(CH_VCE, 0)
        return iceo

    def test_leakage_trend(self) -> List[dict]:
        """
        低压漏电流趋势测试（替代原 BVCEO 击穿测试）。
        虚拟仿真：保留 5-50V 扫描展示完整击穿特性。
        真实硬件：1-5V 扫描，记录漏电流随电压的变化趋势。
        """
        trend = []
        if hasattr(self.driver, 'simulate_bvceo_step'):
            for i_v, vce_step in enumerate(np.arange(5.0, 50.0, 1.0)):
                sim = self.driver.simulate_bvceo_step(vce_step, BUFFER_SIZE)
                raw = sim["leak_data"]
                self._save_raw(f"leak_trend_V{vce_step:.0f}V.npy", raw)
                v_leak = trimmed_mean(raw, TRIM_RATIO, use_sigma=True)
                ic_meas = v_leak / R_LEAK
                trend.append({"vce": vce_step, "ic_leak": ic_meas})
                if ic_meas > 1e-3:
                    break
        else:
            for i_v, vce_step in enumerate(np.arange(1.0, 5.5, 0.5)):
                self.driver.set_voltage(CH_VCE, vce_step)
                _time.sleep(0.3)
                raw = self.driver.read_voltage(0, BUFFER_SIZE)
                self._save_raw(f"leak_trend_V{vce_step:.1f}V.npy", raw)
                v_leak = trimmed_mean(raw, TRIM_RATIO, use_sigma=True)
                ic_meas = v_leak / R_LEAK
                trend.append({"vce": vce_step, "ic_leak": ic_meas})

        self.driver.set_voltage(CH_VCE, 0)
        return trend

    def test_bvceo(self) -> float | None:
        """BVCEO 击穿电压测试。
        虚拟模式：从漏电流趋势外推击穿电压。
        真实模式：硬件限制 5V，无法测量，返回 None。
        """
        trend = self.test_leakage_trend()
        if not trend:
            return None
        if self._is_virtual():
            for point in trend:
                if point["ic_leak"] > 1e-3:
                    return point["vce"]
            return trend[-1]["vce"]
        else:
            return None

    def run_all_tests(self) -> dict:
        """运行全部六项测试，返回汇总结果。
        测试顺序：hFE → 提示改接Kelvin线 → VBE(sat) → 提示改回 → VCE(sat) → ICBO → ICEO → 漏电流
        """
        results = {}

        # 1. hFE
        self._check_stop()
        self._progress(8, "正在测试 hFE...")
        self._log("开始 hFE 测试")
        hfe_data = self.test_hFE()
        avg_hfe = np.mean([r["hfe"] for r in hfe_data])
        status = "PASS" if HFE_MIN <= avg_hfe <= HFE_MAX else "FAIL"
        results["hFE"] = {"value": avg_hfe, "status": status,
                          "detail": hfe_data}
        self._log(f"hFE = {avg_hfe:.1f}, {status}")

        # 2. VBE(sat) — 需要 Kelvin 线接 B-E
        self._check_stop()
        self._prompt_user(
            "请将 ADC CH1 感测线从集电极改接到基极（CH0 留在发射极不动），"
            "然后点击确认继续。")
        self._progress(20, "正在测试 VBE(sat)...")
        self._log("开始 VBE(sat) 测试")
        vbe_sat = self.test_vbe_sat()
        status = "PASS" if vbe_sat < VBESAT_MAX else "FAIL"
        results["VBE(sat)"] = {"value": vbe_sat, "status": status}
        self._log(f"VBE(sat) = {vbe_sat*1e3:.1f} mV, {status}")

        # 3. VCE(sat) — 需要 Kelvin 线接回 C-E
        self._check_stop()
        self._prompt_user(
            "请将 ADC CH1 感测线从基极改回集电极（恢复 C-E Kelvin 接法），"
            "然后点击确认继续。")
        self._progress(35, "正在测试 VCE(sat)...")
        self._log("开始 VCE(sat) 测试")
        vce_sat = self.test_vce_sat()
        status = "PASS" if vce_sat < VCESAT_MAX else "FAIL"
        results["VCE(sat)"] = {"value": vce_sat, "status": status}
        self._log(f"VCE(sat) = {vce_sat:.3f} V, {status}")

        # 4. ICBO
        self._check_stop()
        self._progress(50, "正在测试 ICBO...")
        self._log("开始 ICBO 测试")
        icbo = self.test_icbo()
        status = "PASS" if icbo < ICBO_MAX else "FAIL"
        results["ICBO"] = {"value": icbo, "status": status}
        self._log(f"ICBO = {icbo*1e9:.1f} nA, {status}")

        # 5. ICEO
        self._check_stop()
        self._progress(70, "正在测试 ICEO...")
        self._log("开始 ICEO 测试")
        iceo = self.test_iceo()
        status = "PASS" if iceo < ICEO_MAX else "FAIL"
        results["ICEO"] = {"value": iceo, "status": status}
        self._log(f"ICEO = {iceo*1e9:.1f} nA, {status}")

        # 6. 漏电流趋势测试
        self._check_stop()
        self._progress(90, "正在测试漏电流趋势...")
        self._log("开始漏电流趋势测试")
        trend = self.test_leakage_trend()
        if trend:
            max_leak = max(p["ic_leak"] for p in trend)
            status = "PASS" if max_leak < LEAKAGE_5V_THRESHOLD else "FAIL"
            results["leakage_trend"] = {
                "value": max_leak, "status": status, "detail": trend}
            self._log(f"漏电流趋势: 最大 {max_leak*1e9:.1f} nA, {status}")
        else:
            results["leakage_trend"] = {"value": 0, "status": "N/A"}

        # 7. BVCEO 估算（虚拟模式从漏电流趋势外推，真实模式硬件限制 5V 无法测量）
        if self._is_virtual() and trend:
            bvceo = None
            for point in trend:
                if point["ic_leak"] > 1e-3:
                    bvceo = point["vce"]
                    break
            if bvceo is None:
                bvceo = trend[-1]["vce"]
            status = "PASS" if bvceo >= BVCEO_MIN else "FAIL"
            results["BVCEO"] = {"value": bvceo, "status": status}
            self._log(f"BVCEO = {bvceo:.1f} V (estimated), {status}")

        self._progress(100, "测试完成")
        return results

    # ==================== 特性曲线扫描 ====================

    def _measure_hfe_point(self, ib_target, vce_set):
        """单点 hFE 测量：设置偏置，采集，计算 IB/IC/hFE。"""
        vb_set = ib_target * R_BASE + 0.7
        self.driver.set_voltage(CH_VB, vb_set)
        self.driver.set_voltage(CH_VCE, vce_set)
        if hasattr(self.driver, 'simulate_hfe_measurement'):
            sim = self.driver.simulate_hfe_measurement(
                vb_set, vce_set, BUFFER_SIZE)
            raw_vbe = sim["vbe_data"]
            raw_vce = sim["vce_data"]
        else:
            _time.sleep(0.05)
            raw_vbe = self.driver.read_voltage(CH_VBE, BUFFER_SIZE)
            raw_vce = self.driver.read_voltage(CH_VCE_SENSE, BUFFER_SIZE)
        vbe = trimmed_mean(raw_vbe, TRIM_RATIO, use_sigma=True)
        vce_m = trimmed_mean(raw_vce, TRIM_RATIO, use_sigma=True)
        params = calculate_hfe(vb_set, vbe, vce_set, vce_m,
                               R_BASE, R_SENSE)
        return params

    def sweep_ic_vs_vce(self) -> dict:
        """曲线1: IC-VCE 输出特性族。固定多个 IB，扫描 VCE，测量 IC。"""
        ib_values = [10e-6, 20e-6, 50e-6, 100e-6, 150e-6,
                     200e-6, 250e-6, 300e-6, 400e-6, 500e-6]
        vce_steps = list(np.concatenate([
            np.linspace(0.02, 0.5, 20),
            np.linspace(0.6, 5.0, 40)]))
        ic_matrix = []
        for ib in ib_values:
            self._check_stop()
            ic_row = []
            for vce in vce_steps:
                p = self._measure_hfe_point(ib, vce)
                ic_row.append(p["ic"])
            ic_matrix.append(ic_row)
            self._log(f"IC-VCE 扫描: IB={ib*1e6:.0f}uA 完成")
        self.driver.set_voltage(CH_VB, 0)
        self.driver.set_voltage(CH_VCE, 0)
        return {"ib_values": ib_values, "vce_array": vce_steps,
                "ic_matrix": ic_matrix}

    def sweep_hfe_vs_ic(self, vce_set=5.0) -> dict:
        """曲线2: hFE-IC 特性。固定 VCE，扫描 IB 得到不同 IC。"""
        ib_list = list(np.logspace(-5, np.log10(3e-3), 120))
        ic_arr, hfe_arr = [], []
        for ib in ib_list:
            self._check_stop()
            p = self._measure_hfe_point(ib, vce_set)
            if p["ic"] > 1e-6 and p["hfe"] > 1.0:
                ic_arr.append(p["ic"])
                hfe_arr.append(p["hfe"])
        self.driver.set_voltage(CH_VB, 0)
        self.driver.set_voltage(CH_VCE, 0)
        self._log(f"hFE-IC 扫描完成, {len(ic_arr)} 个有效点")
        return {"ic_array": ic_arr, "hfe_array": hfe_arr}

    def sweep_vbe_sat_vs_ic(self, ratio=10.0) -> dict:
        """曲线3: VBE(sat)-IC。扫描 IC，固定 IC/IB=ratio，测量 VBE。
        真实模式需要 Kelvin 感测线接在 B-E 端（ADC CH1=基极，CH0=发射极）。
        """
        ic_targets = list(np.logspace(-5, -1.3, 50))
        ic_arr, vbe_sat_arr = [], []
        for i_pt, ic_t in enumerate(ic_targets):
            self._check_stop()
            ib_t = ic_t / ratio
            vb_set = ib_t * R_BASE_SAT + 0.7
            vce_force = ic_t * R_SENSE + 0.3
            self.driver.set_voltage(CH_VB, vb_set)
            self.driver.set_voltage(CH_VCE, vce_force)
            if hasattr(self.driver, 'simulate_vbesat_measurement'):
                sim = self.driver.simulate_vbesat_measurement(
                    ic_t, ib_t, BUFFER_SIZE)
                vbe = trimmed_mean(sim["sense_p"], TRIM_RATIO, use_sigma=True) - \
                      trimmed_mean(sim["sense_n"], TRIM_RATIO, use_sigma=True)
            else:
                _time.sleep(0.10)
                raw_sp, raw_sn = self.driver.read_dual_voltage(
                    0, 1, BUFFER_SIZE)
                self._save_raw(f"vbesat_ic_pt{i_pt}_sp.npy", raw_sp)
                self._save_raw(f"vbesat_ic_pt{i_pt}_sn.npy", raw_sn)
                vbe = trimmed_mean(raw_sn, TRIM_RATIO, use_sigma=True) - \
                      trimmed_mean(raw_sp, TRIM_RATIO, use_sigma=True)
            ic_arr.append(ic_t)
            vbe_sat_arr.append(max(vbe, 0.001))
        self.driver.set_voltage(CH_VB, 0)
        self.driver.set_voltage(CH_VCE, 0)
        self._log(f"VBE(sat)-IC 扫描完成, {len(ic_arr)} 点")
        return {"ic_array": ic_arr, "vbe_sat_array": vbe_sat_arr}

    def sweep_vce_sat_vs_ic(self, ratio=10.0) -> dict:
        """曲线4: VCE(sat)-IC。扫描 IC，固定 IC/IB=ratio，测量 VCE(sat)。"""
        ic_targets = list(np.logspace(-5, -1.3, 50))
        ic_arr, vce_sat_arr = [], []
        for i_pt, ic_t in enumerate(ic_targets):
            self._check_stop()
            ib_t = ic_t / ratio
            if hasattr(self.driver, 'simulate_vcesat_measurement'):
                sim = self.driver.simulate_vcesat_measurement(
                    ic_t, ib_t, BUFFER_SIZE)
                vce_sat = trimmed_mean(sim["sense_p"], TRIM_RATIO, use_sigma=True) - \
                          trimmed_mean(sim["sense_n"], TRIM_RATIO, use_sigma=True)
            else:
                vb = ib_t * R_BASE_SAT + 0.7
                vce_force = ic_t * R_SENSE + 0.3
                self.driver.set_voltage(CH_VB, vb)
                self.driver.set_voltage(CH_VCE, vce_force)
                _time.sleep(0.10)
                raw_sp, raw_sn = self.driver.read_dual_voltage(
                    0, 1, BUFFER_SIZE)
                self._save_raw(f"vcesat_ic_pt{i_pt}_sp.npy", raw_sp)
                self._save_raw(f"vcesat_ic_pt{i_pt}_sn.npy", raw_sn)
                vce_sat = trimmed_mean(raw_sn, TRIM_RATIO, use_sigma=True) - \
                          trimmed_mean(raw_sp, TRIM_RATIO, use_sigma=True)
            ic_arr.append(ic_t)
            vce_sat_arr.append(max(vce_sat, 0.001))
        self.driver.set_voltage(CH_VB, 0)
        self.driver.set_voltage(CH_VCE, 0)
        self._log(f"VCE(sat)-IC 扫描完成, {len(ic_arr)} 点")
        return {"ic_array": ic_arr, "vce_sat_array": vce_sat_arr}

    def sweep_ic_vs_vbe(self, vce_set=5.0) -> dict:
        """曲线5: IC-VBE 转移特性。固定 VCE，扫描 VB，测量 IC。"""
        vbe_targets = list(np.linspace(0.7, 2.0, 120))
        vbe_arr, ic_arr = [], []
        self.driver.set_voltage(CH_VCE, vce_set)
        for i_pt, vbe_t in enumerate(vbe_targets):
            self._check_stop()
            self.driver.set_voltage(CH_VB, vbe_t)
            if hasattr(self.driver, 'simulate_hfe_measurement'):
                sim = self.driver.simulate_hfe_measurement(
                    vbe_t, vce_set, BUFFER_SIZE)
                raw_vbe = sim["vbe_data"]
                raw_vce = sim["vce_data"]
            else:
                _time.sleep(0.05)
                raw_vbe = self.driver.read_voltage(CH_VBE, BUFFER_SIZE)
                raw_vce = self.driver.read_voltage(CH_VCE_SENSE, BUFFER_SIZE)
            self._save_raw(f"ic_vbe_pt{i_pt}_ch0.npy", raw_vbe)
            self._save_raw(f"ic_vbe_pt{i_pt}_ch1.npy", raw_vce)
            vbe_m = trimmed_mean(raw_vbe, TRIM_RATIO, use_sigma=True)
            vce_m = trimmed_mean(raw_vce, TRIM_RATIO, use_sigma=True)
            ic = (vce_set - vce_m) / R_SENSE
            vbe_arr.append(vbe_m)
            ic_arr.append(max(ic, 0))
        self.driver.set_voltage(CH_VB, 0)
        self.driver.set_voltage(CH_VCE, 0)
        self._log(f"IC-VBE 扫描完成, {len(vbe_arr)} 点")
        return {"vbe_array": vbe_arr, "ic_array": ic_arr}

    def sweep_vce_vs_ib(self) -> dict:
        """曲线6: VCE-IB 特性。固定多个 IC 目标，扫描 IB，测量 VCE。"""
        ic_targets = [1e-3, 10e-3, 30e-3]
        ib_list = list(np.logspace(-6, -2, 50))
        vce_matrix = []
        for i_ic, ic_t in enumerate(ic_targets):
            self._check_stop()
            vce_row = []
            vce_force = ic_t * R_SENSE + 0.5
            self.driver.set_voltage(CH_VCE, vce_force)
            for i_ib, ib in enumerate(ib_list):
                self._check_stop()
                vb_set = ib * R_BASE_SAT + 0.7
                self.driver.set_voltage(CH_VB, vb_set)
                if hasattr(self.driver, 'simulate_vcesat_measurement'):
                    sim = self.driver.simulate_vcesat_measurement(
                        ic_t, ib, BUFFER_SIZE)
                    vce_sat = trimmed_mean(sim["sense_p"], TRIM_RATIO, use_sigma=True) - \
                              trimmed_mean(sim["sense_n"], TRIM_RATIO, use_sigma=True)
                else:
                    _time.sleep(0.10)
                    raw_sp, raw_sn = self.driver.read_dual_voltage(
                        0, 1, BUFFER_SIZE)
                    self._save_raw(
                        f"vce_ib_ic{int(ic_t*1e3)}mA_pt{i_ib}_sp.npy", raw_sp)
                    self._save_raw(
                        f"vce_ib_ic{int(ic_t*1e3)}mA_pt{i_ib}_sn.npy", raw_sn)
                    vce_sat = trimmed_mean(raw_sn, TRIM_RATIO, use_sigma=True) - \
                              trimmed_mean(raw_sp, TRIM_RATIO, use_sigma=True)
                vce_row.append(min(max(vce_sat, 0.001), 5.0))
            vce_matrix.append(vce_row)
            self._log(f"VCE-IB 扫描: IC={ic_t*1e3:.0f}mA 完成")
        self.driver.set_voltage(CH_VB, 0)
        self.driver.set_voltage(CH_VCE, 0)
        return {"ib_array": ib_list, "ic_targets": ic_targets,
                "vce_matrix": vce_matrix}

    def run_all_sweeps(self) -> dict:
        """执行全部 6 项特性曲线扫描，返回汇总数据。
        扫描顺序：先完成不需要 Kelvin 线的曲线，再提示改接线测 VBE(sat)，
        最后改回 Kelvin 线测 VCE(sat) 和 VCE-IB。
        """
        data = {}

        # 第一阶段：不需要 Kelvin 感测线的曲线
        self._check_stop()
        self._progress(5, "扫描 IC-VCE 输出特性...")
        data["ic_vs_vce"] = self.sweep_ic_vs_vce()

        self._check_stop()
        self._progress(20, "扫描 hFE-IC 特性...")
        data["hfe_vs_ic"] = self.sweep_hfe_vs_ic()

        self._check_stop()
        self._progress(35, "扫描 IC-VBE 转移特性...")
        data["ic_vs_vbe"] = self.sweep_ic_vs_vbe()

        # 第二阶段：改接 Kelvin 线到 B-E，测 VBE(sat)
        self._check_stop()
        self._prompt_user(
            "请将 ADC CH1 感测线从集电极改接到基极（CH0 留在发射极不动），"
            "然后点击确认继续。")
        self._progress(45, "扫描 VBE(sat)-IC...")
        data["vbe_sat_vs_ic"] = self.sweep_vbe_sat_vs_ic()

        # 第三阶段：改回 Kelvin 线到 C-E，测 VCE(sat) 和 VCE-IB
        self._check_stop()
        self._prompt_user(
            "请将 ADC CH1 感测线从基极改回集电极（恢复 C-E Kelvin 接法），"
            "然后点击确认继续。")
        self._progress(60, "扫描 VCE(sat)-IC...")
        data["vce_sat_vs_ic"] = self.sweep_vce_sat_vs_ic()

        self._check_stop()
        self._progress(80, "扫描 VCE-IB 特性...")
        data["vce_vs_ib"] = self.sweep_vce_vs_ib()

        self._progress(100, "曲线扫描完成")
        return data
