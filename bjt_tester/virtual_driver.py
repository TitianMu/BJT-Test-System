"""
virtual_driver.py - 虚拟仪器驱动
基于 Gummel-Poon 模型模拟 2N3904 器件特性，零硬件依赖，可直接运行。
"""
import numpy as np
from typing import List
from driver_base import BaseInstrumentDriver
from bjt_model import GummelPoonBJT, KEC_2N3904

# ===== 负载板电路参数 =====
R_BASE = 1e3        # 基极限流电阻 (Ohm)（与 test_engine 一致）
R_SENSE = 100.0     # 集电极采样电阻 (Ohm)
R_LEAK = 10e6       # 跨阻放大器反馈电阻 (Ohm)
R_PARASITIC = 7.0   # 封装寄生电阻修正 (Ohm)

# ===== 噪声参数 =====
NOISE_RATIO = 0.005  # 信号相对噪声比 0.5%
NOISE_FLOOR = 1e-4   # 噪声底噪 0.1 mV


class VirtualInstrumentDriver(BaseInstrumentDriver):
    """
    虚拟仪器驱动：模拟 Model3 硬件行为。
    所有仪器操作在内存中完成，无需 DLL 和硬件。
    """

    def __init__(self, bjt_params=None):
        self._connected = False
        self._voltages = {}
        self._currents = {}
        self._power = {}
        self._device_name = "VirtualModel3-SIM"
        self._bjt = GummelPoonBJT(bjt_params or KEC_2N3904)
        # 击穿和漏电参数（GP 模型不直接覆盖这些）
        self._bvceo_typ = 42.0
        self._icbo_typ = 15e-9
        self._iceo_typ = 35e-9

    def connect(self) -> bool:
        """模拟设备连接"""
        self._connected = True
        self._voltages = {0: 0.0, 1: 0.0}
        self._currents = {0: 0.0, 1: 0.0}
        return True

    def disconnect(self) -> None:
        """模拟设备断开"""
        self._voltages = {}
        self._currents = {}
        self._power = {}
        self._connected = False

    def set_voltage(self, channel: int, voltage: float) -> None:
        """记录通道电压设定值"""
        self._voltages[channel] = voltage

    def read_voltage(self, channel: int, samples: int = 2048) -> List[float]:
        """
        模拟 ADC 采集：根据当前偏置条件计算理论电压，叠加高斯噪声。
        channel 0: VBE 或跨阻放大器输出
        channel 1: VCE 测量值
        """
        base = self._voltages.get(channel, 0.0)
        noise_std = abs(base) * NOISE_RATIO + NOISE_FLOOR
        noise = np.random.normal(0, noise_std, samples)
        return list(base + noise)

    def set_current_source(self, channel: int, current: float) -> None:
        """记录电流源设定值"""
        self._currents[channel] = current

    def read_current(self, channel: int, samples: int = 2048) -> List[float]:
        """模拟电流采集"""
        base = self._currents.get(channel, 0.0)
        noise_std = abs(base) * NOISE_RATIO + 1e-12
        noise = np.random.normal(0, noise_std, samples)
        return list(base + noise)

    def set_power_supply(self, channel: int, voltage: float,
                         enable: bool = True) -> None:
        """模拟电源控制"""
        self._power[channel] = voltage if enable else 0.0

    def emergency_stop(self) -> None:
        """紧急停止：清空所有输出"""
        self._voltages = {k: 0.0 for k in self._voltages}
        self._currents = {k: 0.0 for k in self._currents}
        self._power = {k: 0.0 for k in self._power}

    def dmm_read_dc_voltage(self) -> float:
        vce = self._voltages.get(1, 0.0)
        return vce + np.random.normal(0, 0.002)

    def dmm_read_diode(self) -> float:
        return 0.65 + np.random.normal(0, 0.005)

    def set_digital_output(self, channel: int, state: bool) -> None:
        self._digital_out = getattr(self, '_digital_out', 0)
        if state:
            self._digital_out |= (1 << channel)
        else:
            self._digital_out &= ~(1 << channel)

    def read_digital_input(self) -> int:
        return getattr(self, '_digital_out', 0)

    def dmm_read_dc_voltage(self) -> float:
        vce = self._voltages.get(1, 0.0)
        return vce + np.random.normal(0, 0.002)

    def dmm_read_diode(self) -> float:
        return 0.65 + np.random.normal(0, 0.005)

    def set_digital_output(self, channel: int, state: bool) -> None:
        if not hasattr(self, '_dio_state'):
            self._dio_state = 0
        if state:
            self._dio_state |= (1 << channel)
        else:
            self._dio_state &= ~(1 << channel)

    def read_digital_input(self) -> int:
        return getattr(self, '_dio_state', 0)

    # ===== 2N3904 物理模型（委托给 GummelPoonBJT）=====
    def _bjt_ic(self, vbe: float, vce: float) -> float:
        """Gummel-Poon 模型计算集电极电流"""
        return self._bjt.compute_ic(vbe, vce)

    def _bjt_vce_sat(self, ic: float, ib: float) -> float:
        """计算饱和压降（含封装寄生电阻修正）"""
        if ib <= 0 or ic <= 0:
            return 0.0
        _, vce_sat = self._bjt.solve_saturation(ic, ib)
        vce_sat += ic * R_PARASITIC
        return vce_sat + np.random.normal(0, 0.003)

    def _bjt_icbo(self, vcb: float) -> float:
        """计算 ICBO"""
        return self._icbo_typ * (1 + vcb / 200) + np.random.normal(0, 0.5e-9)

    def _bjt_iceo(self, vce: float) -> float:
        """计算 ICEO"""
        return self._iceo_typ * (1 + vce / 200) + np.random.normal(0, 1e-9)

    def _bjt_bvceo_current(self, vce: float) -> float:
        """计算 BVCEO 测试中的漏电流"""
        if vce < self._bvceo_typ * 0.8:
            return 1e-9 * (vce / 40)
        elif vce < self._bvceo_typ:
            return 1e-7 * np.exp((vce - self._bvceo_typ * 0.8) / 3)
        else:
            return 1e-3 * np.exp((vce - self._bvceo_typ) / 2)

    # ===== 高级读取方法（供 TestEngine 使用）=====
    def simulate_hfe_measurement(self, vb_set: float, vce_set: float,
                                 samples: int = 2048) -> dict:
        """模拟 hFE 测试的完整采集过程。
        自洽求解：找到 VBE 使得 (vb_set - VBE)/R_BASE = IB(VBE, VCE)。
        """
        from scipy.optimize import brentq

        def circuit_eq(vbe):
            ib_circuit = (vb_set - vbe) / R_BASE
            ib_model = self._bjt.compute_ib(vbe, vce_set)
            return ib_circuit - ib_model

        # 求解自洽 VBE
        try:
            vbe = brentq(circuit_eq, 0.3, min(vb_set - 0.001, 0.95),
                         xtol=1e-10)
        except (ValueError, RuntimeError):
            vbe = 0.65

        ib = (vb_set - vbe) / R_BASE
        ib = max(ib, 1e-12)
        ic = self._bjt.compute_ic(vbe, vce_set)
        ic = max(ic, 0)

        # 模拟 ADC 采集
        vbe_data = list(vbe + np.random.normal(0, 0.002, samples))
        vce_meas = vce_set - ic * R_SENSE
        vce_data = list(vce_meas + np.random.normal(0, 0.003, samples))
        return {"vbe_data": vbe_data, "vce_data": vce_data,
                "ib": ib, "ic": ic}

    def simulate_vcesat_measurement(self, ic_set: float, ib_set: float,
                                    samples: int = 2048) -> dict:
        """模拟 VCE(sat) 测试"""
        vce_sat = self._bjt_vce_sat(ic_set, ib_set)
        sense_p_data = list(vce_sat + np.random.normal(0, 0.001, samples))
        sense_n_data = list(np.random.normal(0, 0.0005, samples))
        return {"sense_p": sense_p_data, "sense_n": sense_n_data}

    def simulate_vbesat_measurement(self, ic_set: float, ib_set: float,
                                    samples: int = 2048) -> dict:
        """模拟 VBE(sat) 测试 — Kelvin 线接在 B-E 上"""
        vbe_sat, _ = self._bjt.solve_saturation(ic_set, ib_set)
        sense_p_data = list(vbe_sat + np.random.normal(0, 0.002, samples))
        sense_n_data = list(np.random.normal(0, 0.0005, samples))
        return {"sense_p": sense_p_data, "sense_n": sense_n_data}

    def simulate_icbo_measurement(self, vcb: float,
                                  samples: int = 2048) -> dict:
        """模拟 ICBO 测试"""
        icbo = self._bjt_icbo(vcb)
        v_out = icbo * R_LEAK
        data = list(v_out + np.random.normal(0, 0.005, samples))
        return {"transimpedance_data": data}

    def simulate_iceo_measurement(self, vce: float,
                                  samples: int = 2048) -> dict:
        """模拟 ICEO 测试"""
        iceo = self._bjt_iceo(vce)
        v_out = iceo * R_LEAK
        data = list(v_out + np.random.normal(0, 0.008, samples))
        return {"transimpedance_data": data}

    def simulate_bvceo_step(self, vce: float,
                            samples: int = 2048) -> dict:
        """模拟 BVCEO 单步测试"""
        ic = self._bjt_bvceo_current(vce)
        v_leak = ic * R_LEAK
        leak_data = list(v_leak + np.random.normal(0, 0.01, samples))
        vce_data = list(vce + np.random.normal(0, 0.05, samples))
        return {"leak_data": leak_data, "vce_data": vce_data, "ic": ic}
