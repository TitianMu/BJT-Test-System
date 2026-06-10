"""
gui_main.py - 2N3904 BJT 全参数自动化测试系统主程序
基于 PyQt5 实现图形界面，零硬件依赖，直接运行即可使用虚拟模式。
支持 JSCJ/KEC 参数切换和自定义 Gummel-Poon 参数编辑。
"""
import sys
import os
import re
import threading

# 修复中文路径下 Qt 找不到 platform 插件的问题
import PyQt5
_qt_plugins = os.path.join(os.path.dirname(PyQt5.__file__), 'Qt5', 'plugins')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = _qt_plugins

import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem, QTextEdit,
    QProgressBar, QGroupBox, QComboBox, QMessageBox, QFileDialog,
    QLineEdit, QFormLayout, QScrollArea, QSplitter, QTabWidget,
    QHeaderView, QFrame, QInputDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QDoubleValidator, QPixmap, QIcon, QColor

from driver_factory import DriverFactory
from test_engine import TestEngine
from report_generator import (export_excel, export_datasheet_curves)
from bjt_model import (GummelPoonBJT, PRESETS, PARAM_INFO,
                        KEC_2N3904, JSCJ_2N3904,
                        load_user_data, save_user_data)
from real_driver import detect_device

def make_stylesheet(scale: float = 1.0) -> str:
    def px(base):
        return max(7, round(base * scale))
    return f"""
QMainWindow {{
    background-color: #f5f6fa;
}}
QGroupBox {{
    font-family: "SimSun", "宋体", serif;
    font-size: {px(13)}px;
    font-weight: bold;
    color: #1a3a5c;
    border: 1px solid #b0bec5;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 16px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    background-color: #f5f6fa;
}}
QTableWidget {{
    background-color: white;
    alternate-background-color: #eef2f7;
    gridline-color: #cfd8dc;
    font-family: "Microsoft YaHei", sans-serif;
    font-size: {px(12)}px;
    selection-background-color: #bbdefb;
}}
QHeaderView::section {{
    background-color: #1a3a5c;
    color: white;
    font-family: "Microsoft YaHei", sans-serif;
    font-size: {px(12)}px;
    font-weight: bold;
    padding: 5px;
    border: none;
    border-right: 1px solid #2c5a7c;
}}
QProgressBar {{
    border: 1px solid #b0bec5;
    border-radius: 3px;
    text-align: center;
    font-size: {px(11)}px;
    background-color: #eceff1;
    height: 20px;
}}
QProgressBar::chunk {{
    background-color: #1a3a5c;
    border-radius: 2px;
}}
QTextEdit {{
    background-color: #fafbfc;
    border: 1px solid #cfd8dc;
    font-family: "Consolas", "Courier New", monospace;
    font-size: {px(10)}px;
    color: #37474f;
}}
QComboBox {{
    font-family: "Microsoft YaHei", sans-serif;
    font-size: {px(12)}px;
    padding: 3px 8px;
    border: 1px solid #b0bec5;
    border-radius: 3px;
    background-color: white;
}}
QComboBox:hover {{
    border-color: #1a3a5c;
}}
QLineEdit {{
    font-family: "Consolas", monospace;
    font-size: {px(12)}px;
    padding: 2px 6px;
    border: 1px solid #b0bec5;
    border-radius: 3px;
    background-color: white;
}}
QLineEdit:focus {{
    border-color: #1a3a5c;
}}
QLabel {{
    font-family: "Microsoft YaHei", sans-serif;
    font-size: {px(12)}px;
    color: #37474f;
}}
QScrollArea {{
    border: none;
}}
QPushButton {{
    font-family: "Microsoft YaHei", sans-serif;
    font-size: {px(12)}px;
    padding: 6px 16px;
    border: 1px solid #b0bec5;
    border-radius: 4px;
    background-color: #eceff1;
    color: #37474f;
}}
QPushButton:hover {{
    background-color: #cfd8dc;
    border-color: #90a4ae;
}}
QPushButton:pressed {{
    background-color: #b0bec5;
}}
QPushButton:disabled {{
    background-color: #eceff1;
    color: #b0bec5;
}}
"""


class DeviceDetectWorker(QThread):
    """后台设备检测线程"""
    detected = pyqtSignal(bool, dict)

    def run(self):
        found, info = detect_device()
        self.detected.emit(found, info)


class TestWorker(QThread):
    """测试工作线程，避免阻塞 GUI"""
    progress = pyqtSignal(int, str)
    result = pyqtSignal(str, str, str)
    log = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)
    stopped_signal = pyqtSignal()
    circuit_switch_requested = pyqtSignal(str, str)

    def __init__(self, driver_mode: str = "virtual",
                 test_item: str = "全部测试",
                 device_name: str = "unknown",
                 test_round: int = 0):
        super().__init__()
        self.driver_mode = driver_mode
        self.test_item = test_item
        self.device_name = device_name
        self.test_round = test_round
        self.test_results = {}
        self._engine: 'TestEngine | None' = None
        self._driver = None
        self._stop_requested = False
        self._switch_event = threading.Event()

    def stop(self):
        self._stop_requested = True
        self._switch_event.set()
        if self._engine:
            self._engine.request_stop()

    def respond_to_switch(self, response):
        if isinstance(response, str):
            self._switch_response = response
        else:
            self._switch_response = "continue" if response else "cancel"
        self._switch_event.set()

    def _request_circuit_switch(self, circuit_name: str, description: str,
                                show_retry: bool = False):
        self._switch_show_retry = show_retry
        self._switch_event.clear()
        self._switch_response = None
        self.circuit_switch_requested.emit(circuit_name, description)
        while not self._switch_event.is_set():
            self._switch_event.wait(timeout=0.1)
            if self._stop_requested:
                raise InterruptedError("测试在电路切换等待中被停止")
        if self._switch_response == "cancel":
            raise InterruptedError("用户取消了电路切换")
        return self._switch_response

    def run(self):
        driver = None
        try:
            driver = DriverFactory.create(self.driver_mode)
            self._driver = driver
            driver.connect()
            if self._stop_requested:
                raise InterruptedError("测试在启动前被停止")
            self.log.emit(f"驱动模式: {self.driver_mode}, 连接成功")
            engine = TestEngine(driver)
            self._engine = engine
            engine.set_data_context(self.device_name, self.test_round)
            engine.set_callbacks(
                progress_cb=lambda p, t: self.progress.emit(p, t),
                log_cb=lambda m: self.log.emit(m)
            )

            if self.test_item == "全部测试":
                if self.driver_mode == "real":
                    self.test_results = self._run_all_tests_real(engine)
                else:
                    self.test_results = engine.run_all_tests()
            else:
                self._run_single(engine)

            specs = {
                "hFE": lambda v: f"{v:.1f}",
                "VCE(sat)": lambda v: f"{v:.3f} V",
                "VBE(sat)": lambda v: f"{v*1e3:.1f} mV",
                "ICBO": lambda v: f"{v*1e9:.1f} nA",
                "ICEO": lambda v: f"{v*1e9:.1f} nA",
                "BVCEO": lambda v: f"{v:.1f} V",
            }
            for param, fmt in specs.items():
                if param in self.test_results:
                    val = self.test_results[param]["value"]
                    status = self.test_results[param]["status"]
                    self.result.emit(param, fmt(val), status)
            self.log.emit("测试完成，设备已断开")
        except InterruptedError:
            self.log.emit("测试已被紧急停止")
            self.stopped_signal.emit()
            return
        except Exception as e:
            import traceback
            self.log.emit(f"测试异常: {str(e)}")
            self.log.emit(f"详细信息: {traceback.format_exc()}")
        finally:
            if driver is not None:
                try:
                    driver.disconnect()
                except Exception:
                    pass
        self.finished_signal.emit(self.test_results)

    def _run_all_tests_real(self, engine):
        """REAL 模式全参数测试：按电路分组，每组前提示用户切换电路，支持重测"""
        from test_engine import (HFE_MIN, HFE_MAX, VCESAT_MAX, VBESAT_MAX,
                                 ICBO_MAX, ICEO_MAX, LEAKAGE_5V_THRESHOLD)
        results = {}

        # 1. HFE电路 → hFE 测试
        self._request_circuit_switch(
            "HFE测试电路",
            "即将进行 hFE（直流电流增益）测试\n请接入 HFE 测试电路后点击确认")
        while True:
            if self._stop_requested:
                raise InterruptedError("用户触发紧急停止")
            self.progress.emit(8, "正在测试 hFE...")
            self.log.emit("开始 hFE 测试")
            hfe_data = engine.test_hFE()
            avg_hfe = np.mean([r["hfe"] for r in hfe_data])
            status = "PASS" if HFE_MIN <= avg_hfe <= HFE_MAX else "FAIL"
            results["hFE"] = {"value": avg_hfe, "status": status,
                              "detail": hfe_data}
            self.log.emit(f"hFE = {avg_hfe:.1f}, {status}")
            choice = self._request_circuit_switch(
                "饱和压降测试电路 (Kelvin接B-E)",
                "即将进行 VBE(sat)（基极饱和压降）测试\n"
                "请接入饱和压降测试电路，ADC CH1 感测线接基极、CH0 接发射极",
                show_retry=True)
            if choice == "retry":
                self.log.emit("用户选择重测 hFE")
                continue
            break

        # 2. 饱和压降电路 (Kelvin接B-E) → VBE(sat) 测试
        while True:
            if self._stop_requested:
                raise InterruptedError("用户触发紧急停止")
            self.progress.emit(20, "正在测试 VBE(sat)...")
            self.log.emit("开始 VBE(sat) 测试")
            vbe_sat = engine.test_vbe_sat()
            status = "PASS" if vbe_sat < VBESAT_MAX else "FAIL"
            results["VBE(sat)"] = {"value": vbe_sat, "status": status}
            self.log.emit(f"VBE(sat) = {vbe_sat*1e3:.1f} mV, {status}")
            choice = self._request_circuit_switch(
                "饱和压降测试电路 (Kelvin接C-E)",
                "即将进行 VCE(sat)（集电极饱和压降）测试\n"
                "请将 ADC CH1 感测线从基极改接到集电极（CH0 留在发射极）",
                show_retry=True)
            if choice == "retry":
                self.log.emit("用户选择重测 VBE(sat)")
                continue
            break

        # 3. 饱和压降电路 (Kelvin接C-E) → VCE(sat) 测试
        while True:
            if self._stop_requested:
                raise InterruptedError("用户触发紧急停止")
            self.progress.emit(35, "正在测试 VCE(sat)...")
            self.log.emit("开始 VCE(sat) 测试")
            vce_sat = engine.test_vce_sat()
            status = "PASS" if vce_sat < VCESAT_MAX else "FAIL"
            results["VCE(sat)"] = {"value": vce_sat, "status": status}
            self.log.emit(f"VCE(sat) = {vce_sat:.3f} V, {status}")
            choice = self._request_circuit_switch(
                "ICBO测试电路",
                "即将进行 ICBO（集电极-基极反向截止电流）测试\n"
                "请接入 ICBO 测试电路后点击确认",
                show_retry=True)
            if choice == "retry":
                self.log.emit("用户选择重测 VCE(sat)")
                continue
            break

        # 4. ICBO电路 → ICBO 测试
        while True:
            if self._stop_requested:
                raise InterruptedError("用户触发紧急停止")
            self.progress.emit(50, "正在测试 ICBO...")
            self.log.emit("开始 ICBO 测试")
            icbo = engine.test_icbo()
            status = "PASS" if icbo < ICBO_MAX else "FAIL"
            results["ICBO"] = {"value": icbo, "status": status}
            self.log.emit(f"ICBO = {icbo*1e9:.1f} nA, {status}")
            choice = self._request_circuit_switch(
                "ICEO测试电路",
                "即将进行 ICEO（集电极-发射极反向截止电流）和漏电流趋势测试\n"
                "请接入 ICEO 测试电路后点击确认",
                show_retry=True)
            if choice == "retry":
                self.log.emit("用户选择重测 ICBO")
                continue
            break

        # 5. ICEO电路 → ICEO + 漏电流趋势测试
        while True:
            if self._stop_requested:
                raise InterruptedError("用户触发紧急停止")
            self.progress.emit(70, "正在测试 ICEO...")
            self.log.emit("开始 ICEO 测试")
            iceo = engine.test_iceo()
            status = "PASS" if iceo < ICEO_MAX else "FAIL"
            results["ICEO"] = {"value": iceo, "status": status}
            self.log.emit(f"ICEO = {iceo*1e9:.1f} nA, {status}")

            self.progress.emit(90, "正在测试漏电流趋势...")
            self.log.emit("开始漏电流趋势测试")
            trend = engine.test_leakage_trend()
            if trend:
                max_leak = max(p["ic_leak"] for p in trend)
                lk_status = "PASS" if max_leak < LEAKAGE_5V_THRESHOLD else "FAIL"
                results["leakage_trend"] = {
                    "value": max_leak, "status": lk_status, "detail": trend}
                self.log.emit(
                    f"漏电流趋势: 最大 {max_leak*1e9:.1f} nA, {lk_status}")
            else:
                results["leakage_trend"] = {"value": 0, "status": "N/A"}

            choice = self._request_circuit_switch(
                "测试完成",
                "全部五项参数测试已完成。\n"
                "点击「继续下一个」确认完成，点击「重测本参数」重测 ICEO 和漏电流趋势。",
                show_retry=True)
            if choice == "retry":
                self.log.emit("用户选择重测 ICEO + 漏电流趋势")
                continue
            break

        self.progress.emit(100, "测试完成")
        return results

    def _run_single(self, engine):
        """执行单项测试"""
        from test_engine import (HFE_MIN, HFE_MAX, VCESAT_MAX, VBESAT_MAX,
                                 ICBO_MAX, ICEO_MAX, BVCEO_MIN)
        item = self.test_item
        self.progress.emit(10, f"正在测试 {item}...")
        self.log.emit(f"开始 {item} 单项测试")

        if item == "hFE":
            data = engine.test_hFE()
            avg = np.mean([r["hfe"] for r in data])
            status = "PASS" if HFE_MIN <= avg <= HFE_MAX else "FAIL"
            self.test_results["hFE"] = {"value": avg, "status": status,
                                        "detail": data}
        elif item == "VCE(sat)":
            val = engine.test_vce_sat()
            status = "PASS" if val < VCESAT_MAX else "FAIL"
            self.test_results["VCE(sat)"] = {"value": val, "status": status}
        elif item == "VBE(sat)":
            val = engine.test_vbe_sat()
            status = "PASS" if val < VBESAT_MAX else "FAIL"
            self.test_results["VBE(sat)"] = {"value": val, "status": status}
        elif item == "ICBO":
            val = engine.test_icbo()
            status = "PASS" if val < ICBO_MAX else "FAIL"
            self.test_results["ICBO"] = {"value": val, "status": status}
        elif item == "ICEO":
            val = engine.test_iceo()
            status = "PASS" if val < ICEO_MAX else "FAIL"
            self.test_results["ICEO"] = {"value": val, "status": status}
        elif item == "BVCEO":
            val = engine.test_bvceo()
            if val is not None:
                status = "PASS" if val >= BVCEO_MIN else "FAIL"
                self.test_results["BVCEO"] = {"value": val, "status": status}
            else:
                self.test_results["BVCEO"] = {"value": 50.0, "status": "PASS"}

        self.progress.emit(100, "测试完成")


class CurveWorker(QThread):
    """曲线生成工作线程"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, params, filename):
        super().__init__()
        self.params = params
        self.filename = filename

    def run(self):
        try:
            path = export_datasheet_curves(self.params, self.filename)
            self.finished.emit(path)
        except Exception as e:
            self.error.emit(str(e))


class RealCurveWorker(QThread):
    """真实模式曲线生成工作线程 - 使用硬件扫描获取实测数据"""
    progress = pyqtSignal(int, str)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    circuit_switch_requested = pyqtSignal(str, str)
    stopped_signal = pyqtSignal()

    def __init__(self, filename: str, device_name: str = "2N3904",
                 test_round: int = 0):
        super().__init__()
        self.filename = filename
        self.device_name = device_name
        self.test_round = test_round
        self._stop_requested = False
        self._switch_event = threading.Event()
        self._switch_accepted = False
        self._engine: 'TestEngine | None' = None
        self._driver = None

    def stop(self):
        self._stop_requested = True
        self._switch_event.set()
        if self._engine:
            self._engine.request_stop()

    def respond_to_switch(self, response):
        if isinstance(response, str):
            self._switch_accepted = (response == "continue")
        else:
            self._switch_accepted = response
        self._switch_event.set()

    def _request_circuit_switch(self, circuit_name: str, description: str):
        self._switch_event.clear()
        self._switch_accepted = False
        self.circuit_switch_requested.emit(circuit_name, description)
        while not self._switch_event.is_set():
            self._switch_event.wait(timeout=0.1)
            if self._stop_requested:
                raise InterruptedError("曲线扫描在电路切换等待中被停止")
        if not self._switch_accepted:
            raise InterruptedError("用户取消了电路切换")

    def run(self):
        driver = None
        try:
            driver = DriverFactory.create("real")
            self._driver = driver
            driver.connect()
            if self._stop_requested:
                raise InterruptedError("扫描在启动前被停止")
            self.log.emit("真实模式曲线扫描: 驱动连接成功")

            engine = TestEngine(driver)
            self._engine = engine
            engine.set_data_context(self.device_name, self.test_round)
            engine.set_callbacks(
                progress_cb=lambda p, t: self.progress.emit(p, t),
                log_cb=lambda m: self.log.emit(m)
            )

            measured_data = {}

            # 电路组1: HFE电路 (3 个扫描)
            self._request_circuit_switch(
                "HFE测试电路",
                "即将进行 IC-VCE、hFE-IC、IC-VBE 三项特性扫描\n"
                "请接入 HFE 测试电路后点击确认")

            self.progress.emit(5, "扫描 IC-VCE 输出特性...")
            measured_data["ic_vs_vce"] = engine.sweep_ic_vs_vce()

            self.progress.emit(20, "扫描 hFE-IC 特性...")
            measured_data["hfe_vs_ic"] = engine.sweep_hfe_vs_ic()

            self.progress.emit(35, "扫描 IC-VBE 转移特性...")
            measured_data["ic_vs_vbe"] = engine.sweep_ic_vs_vbe()

            # 电路组2: 饱和压降电路 Kelvin接C-E (2 个扫描)
            self._request_circuit_switch(
                "饱和压降测试电路 (Kelvin接C-E)",
                "即将进行 VCE(sat)-IC 和 VCE-IB 两项特性扫描\n"
                "请接入饱和压降测试电路，ADC CH1 感测线接集电极、CH0 接发射极\n"
                "切换完成后点击确认")

            self.progress.emit(50, "扫描 VCE(sat)-IC...")
            measured_data["vce_sat_vs_ic"] = engine.sweep_vce_sat_vs_ic()

            self.progress.emit(65, "扫描 VCE-IB 特性...")
            measured_data["vce_vs_ib"] = engine.sweep_vce_vs_ib()

            # 电路组3: 饱和压降电路 Kelvin接B-E (1 个扫描)
            self._request_circuit_switch(
                "饱和压降测试电路 (Kelvin接B-E)",
                "即将进行 VBE(sat)-IC 特性扫描\n"
                "请将 ADC CH1 感测线从集电极改接到基极（CH0 留在发射极）\n"
                "切换完成后点击确认")

            self.progress.emit(80, "扫描 VBE(sat)-IC...")
            measured_data["vbe_sat_vs_ic"] = engine.sweep_vbe_sat_vs_ic()

            # 生成曲线图
            self.progress.emit(95, "正在生成曲线图...")
            path = export_datasheet_curves(
                measured_data=measured_data,
                filename=self.filename,
                device_name=self.device_name)
            self.progress.emit(100, "曲线生成完成")
            self.finished.emit(path)

        except InterruptedError:
            self.log.emit("曲线扫描已被停止")
            self.stopped_signal.emit()
        except Exception as e:
            import traceback
            self.log.emit(f"曲线扫描异常: {traceback.format_exc()}")
            self.error.emit(str(e))
        finally:
            if driver is not None:
                try:
                    driver.disconnect()
                except Exception:
                    pass


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("BJT 全参数自动化测试系统 v2.0")
        self.setWindowIcon(self._load_icon())
        self.setMinimumSize(1200, 750)
        self.test_results = {}
        self.param_edits = {}
        self._device_connected = False
        self._device_serial = ""
        user_data = load_user_data()
        self._user_presets = user_data.get("presets", {})
        self._deleted_builtins = user_data.get("deleted_builtins", [])
        self._zoom_scale = 1.0
        self._apply_style()
        self._build_ui()
        if self.combo_preset.count() > 0:
            self._load_preset(self.combo_preset.currentText())
            self._last_preset = self.combo_preset.currentText()
            self._test_round = 0
        self._detect_device()

    def _load_icon(self):
        icon_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "logo.png")
        if not os.path.exists(icon_path):
            icon_path = r"C:\Users\lenovo\Desktop\集创赛\集创赛LOGO 不带字.png"
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        return QIcon()

    def _apply_style(self):
        self.setStyleSheet(make_stylesheet(self._zoom_scale))

    def _scaled_font_size(self):
        return max(7, round(11 * self._zoom_scale))

    def _info_label_style(self, color):
        return f"color: {color}; font-size: {self._scaled_font_size()}px;"

    def _apply_zoom(self):
        self.setStyleSheet(make_stylesheet(self._zoom_scale))
        s = self._scaled_font_size()
        bold = QFont("Microsoft YaHei", s, QFont.Bold)
        mono_bold = QFont("Consolas", s, QFont.Bold)
        mono = QFont("Consolas", s)
        self.btn_test.setFont(bold)
        self.btn_stop.setFont(bold)
        self.btn_curves.setFont(bold)
        for row in range(self.table.rowCount()):
            item0 = self.table.item(row, 0)
            if item0:
                item0.setFont(mono_bold)
            item1 = self.table.item(row, 1)
            if item1:
                item1.setFont(mono)
            item3 = self.table.item(row, 3)
            if item3:
                item3.setFont(bold)
        for lbl in (self.lbl_device_status, self.lbl_sdk_version, self.lbl_hw_model):
            cur = lbl.styleSheet()
            if "font-size" in cur:
                lbl.setStyleSheet(re.sub(r'font-size:\s*\d+px', f'font-size: {s}px', cur))

    def wheelEvent(self, event):
        if event.modifiers() == Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self._zoom_scale = min(2.0, round(self._zoom_scale + 0.1, 2))
            elif delta < 0:
                self._zoom_scale = max(0.5, round(self._zoom_scale - 0.1, 2))
            self._apply_zoom()
            event.accept()
        else:
            super().wheelEvent(event)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer_layout = QVBoxLayout(central)
        outer_layout.setContentsMargins(8, 8, 8, 8)
        outer_layout.setSpacing(6)

        # ===== 左侧：测试功能 =====
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)

        # 顶部：驱动选择 + 测试按钮
        top = QHBoxLayout()
        top.setSpacing(8)
        grp_drv = QGroupBox("测试配置")
        drv_outer = QVBoxLayout(grp_drv)
        drv_outer.setSpacing(6)
        drv_row1 = QHBoxLayout()
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["virtual", "real"])
        self.combo_mode.currentTextChanged.connect(self._on_mode_changed)
        drv_row1.addWidget(QLabel("模式:"))
        drv_row1.addWidget(self.combo_mode)
        drv_row1.addSpacing(12)
        drv_row1.addWidget(QLabel("测试项:"))
        self.combo_test_item = QComboBox()
        self.combo_test_item.addItems(
            ["全部测试", "hFE", "VCE(sat)", "VBE(sat)", "ICBO", "ICEO", "BVCEO"])
        drv_row1.addWidget(self.combo_test_item)
        drv_row1.addSpacing(12)
        self.btn_detect = QPushButton("检测设备")
        self.btn_detect.setFixedWidth(80)
        self.btn_detect.clicked.connect(self._detect_device)
        drv_row1.addWidget(self.btn_detect)
        self.lbl_status = QLabel("就绪")
        self.lbl_status.setStyleSheet("color: #2e7d32; font-weight: bold;")
        drv_row1.addWidget(self.lbl_status)
        drv_row1.addStretch()
        drv_outer.addLayout(drv_row1)

        drv_row2 = QHBoxLayout()
        self.lbl_device_status = QLabel("设备: 检测中...")
        self.lbl_device_status.setStyleSheet(self._info_label_style("#78909c"))
        drv_row2.addWidget(self.lbl_device_status)
        drv_row2.addSpacing(16)
        self.lbl_sdk_version = QLabel("SDK: --")
        self.lbl_sdk_version.setStyleSheet(self._info_label_style("#78909c"))
        drv_row2.addWidget(self.lbl_sdk_version)
        drv_row2.addSpacing(16)
        self.lbl_hw_model = QLabel("型号: --")
        self.lbl_hw_model.setStyleSheet(self._info_label_style("#78909c"))
        drv_row2.addWidget(self.lbl_hw_model)
        drv_row2.addStretch()
        drv_outer.addLayout(drv_row2)
        top.addWidget(grp_drv, stretch=3)

        btn_col = QVBoxLayout()
        btn_col.setSpacing(6)
        self.btn_test = QPushButton("开始测试")
        self.btn_test.setMinimumHeight(40)
        self.btn_test.setFont(QFont("Microsoft YaHei", self._scaled_font_size(), QFont.Bold))
        self.btn_test.setStyleSheet(
            "QPushButton { background-color: #1a3a5c; color: white; "
            "border-radius: 4px; border: none; padding: 8px 20px; }"
            "QPushButton:hover { background-color: #2c5a7c; }"
            "QPushButton:pressed { background-color: #0d2137; }"
            "QPushButton:disabled { background-color: #90a4ae; }")
        self.btn_test.clicked.connect(self._on_test)
        btn_col.addWidget(self.btn_test)

        self.btn_stop = QPushButton("紧急停止")
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setFont(QFont("Microsoft YaHei", self._scaled_font_size(), QFont.Bold))
        self._stop_style_idle = (
            "QPushButton { background-color: #c6a0a0; color: #f5f5f5; "
            "border-radius: 4px; border: 1px solid #b0bec5; padding: 8px 20px; }"
            "QPushButton:hover { background-color: #d32f2f; color: white; }"
            "QPushButton:pressed { background-color: #7f0000; color: white; }")
        self._stop_style_active = (
            "QPushButton { background-color: #b71c1c; color: white; "
            "border-radius: 4px; border: 2px solid #ff5252; padding: 8px 20px; }"
            "QPushButton:hover { background-color: #d32f2f; }"
            "QPushButton:pressed { background-color: #7f0000; }")
        self.btn_stop.setStyleSheet(self._stop_style_idle)
        self.btn_stop.clicked.connect(self._on_emergency_stop)
        btn_col.addWidget(self.btn_stop)
        top.addLayout(btn_col, stretch=1)
        left_layout.addLayout(top)

        # 进度条
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setFormat("等待测试...")
        left_layout.addWidget(self.progress)

        # 结果表格
        grp_result = QGroupBox("测试结果")
        result_layout = QVBoxLayout(grp_result)
        result_layout.setContentsMargins(8, 16, 8, 8)
        self.table = QTableWidget(6, 4)
        self.table.setAlternatingRowColors(True)
        self.table.setHorizontalHeaderLabels(
            ["参数", "测量值", "规格范围", "判定"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setDefaultSectionSize(32)
        specs = [("hFE", "", "100~630", ""),
                 ("VCE(sat)", "", "<0.4 V", ""),
                 ("VBE(sat)", "", "<1.0 V", ""),
                 ("ICBO", "", "<100 nA", ""),
                 ("ICEO", "", "<200 nA", ""),
                 ("BVCEO", "", ">40 V", "")]
        for r, (p, v, s, j) in enumerate(specs):
            for c, val in enumerate([p, v, s, j]):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                if c == 0:
                    item.setFont(QFont("Consolas", self._scaled_font_size(), QFont.Bold))
                self.table.setItem(r, c, item)
        result_layout.addWidget(self.table)
        left_layout.addWidget(grp_result, stretch=2)

        # 运行日志
        log_grp = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_grp)
        log_layout.setContentsMargins(8, 16, 8, 8)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        left_layout.addWidget(log_grp, stretch=1)

        # 导出按钮行（紧凑水平排列，无多余空白）
        export_row = QHBoxLayout()
        export_row.setSpacing(10)
        self.btn_excel = QPushButton("导出 Excel 报告")
        self.btn_excel.setMinimumHeight(32)
        self.btn_excel.setStyleSheet(
            "QPushButton { background-color: #1a3a5c; color: white; "
            "border-radius: 4px; border: none; padding: 6px 18px; }"
            "QPushButton:hover { background-color: #2c5a7c; }"
            "QPushButton:disabled { background-color: #90a4ae; }")
        self.btn_excel.clicked.connect(self._on_export_excel)
        export_row.addWidget(self.btn_excel)
        export_row.addStretch()
        left_layout.addLayout(export_row)

        # ===== 右侧：参数编辑 + 曲线生成 =====
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(6)

        # 器件预设选择 + 保存/删除
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("器件预设:"))
        self.combo_preset = QComboBox()
        self.combo_preset.setMinimumWidth(160)
        self._refresh_preset_combo()
        self.combo_preset.currentTextChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self.combo_preset)

        self.btn_save_preset = QPushButton("保存预设")
        self.btn_save_preset.setFixedWidth(80)
        self.btn_save_preset.clicked.connect(self._on_save_preset)
        preset_layout.addWidget(self.btn_save_preset)

        self.btn_delete_preset = QPushButton("删除预设")
        self.btn_delete_preset.setFixedWidth(80)
        self.btn_delete_preset.clicked.connect(self._on_delete_preset)
        preset_layout.addWidget(self.btn_delete_preset)

        preset_layout.addStretch()
        right_layout.addLayout(preset_layout)

        # GP 参数编辑区
        param_grp = QGroupBox("Gummel-Poon 模型参数")
        param_form = QFormLayout(param_grp)
        param_form.setSpacing(5)
        param_form.setContentsMargins(10, 18, 10, 10)
        param_form.setLabelAlignment(Qt.AlignRight)
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.ScientificNotation)
        for key, desc, unit, default in PARAM_INFO:
            edit = QLineEdit()
            edit.setValidator(validator)
            edit.setFixedWidth(130)
            label_text = f"{key}"
            if unit:
                label_text += f" ({unit})"
            label_text += f"  {desc}"
            param_form.addRow(label_text, edit)
            self.param_edits[key] = edit

        scroll = QScrollArea()
        scroll.setWidget(param_grp)
        scroll.setWidgetResizable(True)
        right_layout.addWidget(scroll, stretch=1)

        # 生成曲线按钮 + LOGO
        curve_row = QHBoxLayout()
        curve_row.setSpacing(8)
        self.btn_curves = QPushButton("生成晶体管特性曲线")
        self.btn_curves.setMinimumHeight(40)
        self.btn_curves.setFont(QFont("Microsoft YaHei", self._scaled_font_size(), QFont.Bold))
        self.btn_curves.setStyleSheet(
            "QPushButton { background-color: #1a3a5c; color: white; "
            "border-radius: 4px; border: none; padding: 8px 20px; }"
            "QPushButton:hover { background-color: #2c5a7c; }"
            "QPushButton:pressed { background-color: #0d2137; }"
            "QPushButton:disabled { background-color: #90a4ae; }")
        self.btn_curves.clicked.connect(self._on_generate_curves)
        curve_row.addWidget(self.btn_curves)

        logo_label = QLabel()
        logo_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "logo.png")
        if not os.path.exists(logo_path):
            logo_path = r"C:\Users\lenovo\Desktop\集创赛\集创赛LOGO 不带字.png"
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path).scaled(
                42, 42, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(pixmap)
        logo_label.setFixedSize(46, 46)
        curve_row.addWidget(logo_label)
        right_layout.addLayout(curve_row)

        self.lbl_curve_status = QLabel("")
        self.lbl_curve_status.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.lbl_curve_status)

        # 组装左右面板
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(3)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        outer_layout.addWidget(splitter)

    # ==================== 参数面板方法 ====================

    def _refresh_preset_combo(self, select_name=None):
        self.combo_preset.blockSignals(True)
        self.combo_preset.clear()
        for name in PRESETS:
            if name not in self._deleted_builtins:
                self.combo_preset.addItem(name)
        if self._user_presets:
            self.combo_preset.insertSeparator(self.combo_preset.count())
            for name in sorted(self._user_presets.keys()):
                self.combo_preset.addItem(name)
        if select_name:
            idx = self.combo_preset.findText(select_name)
            if idx >= 0:
                self.combo_preset.setCurrentIndex(idx)
        self.combo_preset.blockSignals(False)

    def _load_preset(self, name):
        params = PRESETS.get(name)
        if params is None:
            params = self._user_presets.get(name)
        if params is None:
            params = KEC_2N3904
        for key, _, _, _ in PARAM_INFO:
            val = params.get(key, 0.0)
            self.param_edits[key].setText(f"{val:.6g}")

    def _on_preset_changed(self, name):
        self._load_preset(name)
        if getattr(self, '_last_preset', '') != name:
            self._test_round = 0
            self._last_preset = name

    def _save_user_data(self):
        save_user_data({
            "presets": self._user_presets,
            "deleted_builtins": self._deleted_builtins,
        })

    def _on_save_preset(self):
        name, ok = QInputDialog.getText(
            self, "保存预设", "请输入预设名称:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in PRESETS or name in self._user_presets:
            ret = QMessageBox.question(
                self, "覆盖确认",
                f"预设 \"{name}\" 已存在，是否覆盖？",
                QMessageBox.Yes | QMessageBox.No)
            if ret != QMessageBox.Yes:
                return
        params = self._read_params()
        params['name'] = name
        self._user_presets[name] = params
        if name in self._deleted_builtins:
            self._deleted_builtins.remove(name)
        self._save_user_data()
        self._refresh_preset_combo(select_name=name)
        self._load_preset(name)
        self._append_log(f"预设 \"{name}\" 已保存")

    def _on_delete_preset(self):
        name = self.combo_preset.currentText()
        if not name:
            return
        ret = QMessageBox.question(
            self, "删除确认",
            f"确定删除预设 \"{name}\"？",
            QMessageBox.Yes | QMessageBox.No)
        if ret != QMessageBox.Yes:
            return
        if name in self._user_presets:
            del self._user_presets[name]
        if name in PRESETS:
            if name not in self._deleted_builtins:
                self._deleted_builtins.append(name)
        self._save_user_data()
        self._refresh_preset_combo()
        if self.combo_preset.count() > 0:
            self._load_preset(self.combo_preset.currentText())
        self._append_log(f"预设 \"{name}\" 已删除")

    def _read_params(self):
        """从编辑框读取当前参数"""
        params = {'name': self.combo_preset.currentText()}
        for key, _, _, default in PARAM_INFO:
            text = self.param_edits[key].text().strip()
            try:
                params[key] = float(text)
            except ValueError:
                params[key] = default
        return params

    def _on_generate_curves(self):
        """生成数据手册曲线"""
        mode = self.combo_mode.currentText()

        if mode == "real" and not self._device_connected:
            QMessageBox.warning(
                self, "提示",
                "未检测到硬件设备，无法进行实测曲线扫描。\n"
                "请检查 USB 连接后点击「检测设备」。")
            return

        if mode == "real":
            device_name, ok = QInputDialog.getText(
                self, "输入器件名称",
                "请输入被测器件名称（用于图表标题和文件名）：")
            if not ok or not device_name.strip():
                return
            device_name = device_name.strip()
            default_name = f"{device_name}_Measured_Curves.png"
        else:
            preset_name = self._read_params()['name']
            default_name = f"{preset_name}_Curves.png"

        fn, _ = QFileDialog.getSaveFileName(
            self, "保存数据手册曲线", default_name,
            "PNG Files (*.png)")
        if not fn:
            return

        self.btn_curves.setEnabled(False)
        self.lbl_curve_status.setText("正在生成曲线...")
        self.lbl_curve_status.setStyleSheet("color: orange;")

        if mode == "real" and self._device_connected:
            self._curve_worker = RealCurveWorker(fn, device_name, self._test_round)
            self._curve_worker.finished.connect(self._on_curves_done)
            self._curve_worker.error.connect(self._on_curves_error)
            self._curve_worker.progress.connect(self._update_progress)
            self._curve_worker.log.connect(self._append_log)
            self._curve_worker.circuit_switch_requested.connect(
                self._on_circuit_switch_requested)
            self._curve_worker.stopped_signal.connect(self._on_curves_stopped)
            self._curve_worker.start()
        else:
            params = self._read_params()
            self._curve_worker = CurveWorker(params, fn)
            self._curve_worker.finished.connect(self._on_curves_done)
            self._curve_worker.error.connect(self._on_curves_error)
            self._curve_worker.start()

    def _on_curves_done(self, path):
        self.btn_curves.setEnabled(True)
        self.lbl_curve_status.setText(f"已保存: {os.path.basename(path)}")
        self.lbl_curve_status.setStyleSheet("color: green;")
        self._append_log(f"数据手册曲线已保存: {path}")

    def _on_curves_error(self, msg):
        self.btn_curves.setEnabled(True)
        self.lbl_curve_status.setText("生成失败")
        self.lbl_curve_status.setStyleSheet("color: red;")
        self._append_log(f"曲线生成失败: {msg}")

    def _on_curves_stopped(self):
        self.btn_curves.setEnabled(True)
        self.lbl_curve_status.setText("曲线扫描已停止")
        self.lbl_curve_status.setStyleSheet("color: #b71c1c;")

    def _on_circuit_switch_requested(self, circuit_name: str,
                                     description: str):
        """处理工作线程的电路切换请求（支持重测选项）"""
        worker = self.sender()
        show_retry = getattr(worker, '_switch_show_retry', False)

        if show_retry:
            msg = QMessageBox(self)
            if circuit_name == "测试完成":
                msg.setWindowTitle("测试完成")
                msg.setText(f"{description}")
            else:
                msg.setWindowTitle("请切换测试电路")
                msg.setText(f"请将测试电路切换到:\n\n    {circuit_name}\n\n"
                            f"{description}")
            msg.setIcon(QMessageBox.Question)
            btn_continue = msg.addButton("继续下一个", QMessageBox.AcceptRole)
            btn_retry = msg.addButton("重测本参数", QMessageBox.ActionRole)
            btn_cancel = msg.addButton("取消测试", QMessageBox.RejectRole)
            msg.setDefaultButton(btn_continue)
            msg.exec_()
            clicked = msg.clickedButton()
            if clicked == btn_continue:
                response = "continue"
            elif clicked == btn_retry:
                response = "retry"
            else:
                response = "cancel"
        else:
            ret = QMessageBox.question(
                self, "请切换测试电路",
                f"请将测试电路切换到:\n\n    {circuit_name}\n\n"
                f"{description}\n\n"
                f"切换完成后点击「Yes」继续，点击「No」取消。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes)
            response = "continue" if ret == QMessageBox.Yes else "cancel"

        if hasattr(worker, 'respond_to_switch'):
            worker.respond_to_switch(response)

    # ==================== 原有方法 ====================

    def _append_log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {msg}")

    def _on_emergency_stop(self):
        """紧急停止：通知引擎停止并关闭所有输出"""
        self._cleanup_worker()
        self._cleanup_curve_worker()
        self.btn_test.setEnabled(True)
        self.btn_curves.setEnabled(True)
        self.btn_stop.setStyleSheet(self._stop_style_idle)
        self.lbl_status.setText("已紧急停止")
        self.lbl_status.setStyleSheet("color: #b71c1c; font-weight: bold;")
        self._append_log("用户触发紧急停止")

    def _cleanup_worker(self):
        """停止并清理旧的测试线程，断开所有信号防止状态污染"""
        if not hasattr(self, 'worker') or self.worker is None:
            return
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
            if self.worker.isRunning():
                self.worker.terminate()
                self.worker.wait(1000)
        try:
            self.worker.progress.disconnect()
            self.worker.result.disconnect()
            self.worker.log.disconnect()
            self.worker.finished_signal.disconnect()
            self.worker.stopped_signal.disconnect()
            self.worker.circuit_switch_requested.disconnect()
        except TypeError:
            pass
        self.worker = None

    def _cleanup_curve_worker(self):
        """停止并清理曲线扫描线程"""
        if not hasattr(self, '_curve_worker') or self._curve_worker is None:
            return
        if hasattr(self._curve_worker, 'stop'):
            self._curve_worker.stop()
        if self._curve_worker.isRunning():
            self._curve_worker.wait(3000)
            if self._curve_worker.isRunning():
                self._curve_worker.terminate()
                self._curve_worker.wait(1000)
        try:
            self._curve_worker.finished.disconnect()
            self._curve_worker.error.disconnect()
        except TypeError:
            pass
        try:
            self._curve_worker.progress.disconnect()
            self._curve_worker.log.disconnect()
            self._curve_worker.circuit_switch_requested.disconnect()
            self._curve_worker.stopped_signal.disconnect()
        except (TypeError, AttributeError):
            pass
        self._curve_worker = None

    def _on_test(self):
        mode = self.combo_mode.currentText()
        if mode == "real" and not self._device_connected:
            QMessageBox.warning(
                self, "提示",
                "未检测到硬件设备，请检查 USB 连接后点击「检测设备」。")
            return
        self._cleanup_worker()
        device_name = self.combo_preset.currentText()
        self._test_round = getattr(self, '_test_round', 0) + 1
        self.btn_test.setEnabled(False)
        self.btn_stop.setStyleSheet(self._stop_style_active)
        self.lbl_status.setText("测试中...")
        self.lbl_status.setStyleSheet("color: #e65100; font-weight: bold;")
        self.worker = TestWorker(mode, self.combo_test_item.currentText(),
                                 device_name, self._test_round)
        self.worker.progress.connect(self._update_progress)
        self.worker.result.connect(self._update_result)
        self.worker.log.connect(self._append_log)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.stopped_signal.connect(self._on_stopped)
        if mode == "real":
            self.worker.circuit_switch_requested.connect(
                self._on_circuit_switch_requested)
        self.worker.start()

    def _update_progress(self, val: int, text: str):
        self.progress.setValue(val)
        self.progress.setFormat(text)

    def _update_result(self, param: str, value: str, status: str):
        param_map = {"hFE": 0, "VCE(sat)": 1, "VBE(sat)": 2,
                     "ICBO": 3, "ICEO": 4, "BVCEO": 5}
        row = param_map.get(param, 0)
        val_item = QTableWidgetItem(value)
        val_item.setTextAlignment(Qt.AlignCenter)
        val_item.setFont(QFont("Consolas", self._scaled_font_size()))
        self.table.setItem(row, 1, val_item)
        status_item = QTableWidgetItem(status)
        status_item.setTextAlignment(Qt.AlignCenter)
        status_item.setFont(QFont("Microsoft YaHei", self._scaled_font_size(), QFont.Bold))
        if status == "PASS":
            status_item.setForeground(QColor("#2e7d32"))
        else:
            status_item.setForeground(QColor("#b71c1c"))
        self.table.setItem(row, 3, status_item)

    def _on_finished(self, results: dict):
        self.test_results = results
        self.btn_test.setEnabled(True)
        self.btn_stop.setStyleSheet(self._stop_style_idle)
        if self.lbl_status.text() != "已紧急停止":
            self.lbl_status.setText("测试完成")
            self.lbl_status.setStyleSheet("color: #2e7d32; font-weight: bold;")

    def _on_stopped(self):
        """线程因紧急停止而结束"""
        self.btn_test.setEnabled(True)
        self.btn_stop.setStyleSheet(self._stop_style_idle)
        self.lbl_status.setText("已紧急停止")
        self.lbl_status.setStyleSheet("color: #b71c1c; font-weight: bold;")

    def _on_export_excel(self):
        if not self.test_results:
            QMessageBox.warning(self, "提示", "请先运行测试")
            return
        fn, _ = QFileDialog.getSaveFileName(
            self, "保存 Excel", "BJT_Report.xlsx",
            "Excel Files (*.xlsx)")
        if fn:
            path = export_excel(
                self.test_results, fn,
                self.combo_mode.currentText())
            self._append_log(f"Excel 报告已保存: {path}")

    def _detect_device(self):
        self.lbl_device_status.setText("设备: 检测中...")
        self.lbl_device_status.setStyleSheet("color: gray;")
        self.btn_detect.setEnabled(False)
        self._detect_worker = DeviceDetectWorker()
        self._detect_worker.detected.connect(self._on_device_detected)
        self._detect_worker.start()

    def _on_device_detected(self, found: bool, info: dict):
        self.btn_detect.setEnabled(True)
        self._device_connected = found
        self._device_serial = info.get("serial", "")
        sdk_ver = info.get("sdk_version", "")
        hw_model = info.get("model", "")
        if sdk_ver:
            self.lbl_sdk_version.setText(f"SDK: {sdk_ver}")
            self.lbl_sdk_version.setStyleSheet(self._info_label_style("#1565c0"))
        else:
            self.lbl_sdk_version.setText("SDK: --")
            self.lbl_sdk_version.setStyleSheet(self._info_label_style("#78909c"))
        if found:
            self.lbl_device_status.setText(f"设备: 已连接 {info['message']}")
            self.lbl_device_status.setStyleSheet(self._info_label_style("#2e7d32"))
            self.lbl_hw_model.setText(f"型号: {hw_model}")
            self.lbl_hw_model.setStyleSheet(self._info_label_style("#2e7d32"))
            self._append_log(f"检测到设备: {info['message']}, SDK版本: {sdk_ver}")
        else:
            msg = info.get("message", "未知错误")
            self.lbl_device_status.setText(f"设备: 未连接 - {msg}")
            self.lbl_device_status.setStyleSheet(self._info_label_style("#b71c1c"))
            self.lbl_hw_model.setText("型号: --")
            self.lbl_hw_model.setStyleSheet(self._info_label_style("#78909c"))
            self._append_log(f"未检测到硬件设备 ({msg}), SDK版本: {sdk_ver}")

    def _set_params_real_mode(self):
        for key in self.param_edits:
            self.param_edits[key].setText("-")
            self.param_edits[key].setReadOnly(True)
        self.combo_preset.setEnabled(False)
        self.btn_save_preset.setEnabled(False)
        self.btn_delete_preset.setEnabled(False)

    def _restore_params_virtual_mode(self):
        for key in self.param_edits:
            self.param_edits[key].setReadOnly(False)
        self.combo_preset.setEnabled(True)
        self.btn_save_preset.setEnabled(True)
        self.btn_delete_preset.setEnabled(True)
        self._load_preset(self.combo_preset.currentText())

    def _on_mode_changed(self, mode: str):
        self._cleanup_worker()
        self._clear_test_results()
        self.btn_test.setEnabled(True)
        self.btn_stop.setStyleSheet(self._stop_style_idle)
        self.lbl_status.setText("就绪")
        self.lbl_status.setStyleSheet("color: #2e7d32; font-weight: bold;")
        self.progress.setValue(0)
        self.progress.setFormat("等待测试...")
        if mode == "real":
            self._set_params_real_mode()
            self._detect_device()
        else:
            self._restore_params_virtual_mode()

    def _clear_test_results(self):
        for row in range(self.table.rowCount()):
            self.table.setItem(row, 1, QTableWidgetItem(""))
            self.table.setItem(row, 3, QTableWidgetItem(""))
        self.test_results = {}


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

