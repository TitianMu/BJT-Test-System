"""
real_driver.py - 真实仪器驱动
基于雨骤 IP-SDK 3.2 的 pyRD 库，驱动 Raindrop Model-3 硬件。
"""
import sys
import os
import pathlib
import time
from typing import List
from driver_base import BaseInstrumentDriver

_bjt_dir = pathlib.Path(os.path.abspath(__file__)).parent
_sdk_src = str((_bjt_dir / ".." / ".." / "IPSDK3.2" / "IP-SDK" / "Python" / "src").resolve())
if _sdk_src not in sys.path:
    sys.path.insert(0, _sdk_src)

R_BASE = 1e3
R_SENSE = 100.0
ADC_TIMEOUT_S = 5.0
ADC_POLL_INTERVAL = 0.01


class RealInstrumentDriver(BaseInstrumentDriver):

    def __init__(self):
        self._rd = None
        self._connected = False
        self._save_raw = False
        self._raw_counter = 0

    def enable_raw_saving(self, enable: bool = True):
        """启用/禁用驱动层原始 ADC 数据保存（调试用）。"""
        self._save_raw = enable
        self._raw_counter = 0

    def connect(self) -> bool:
        from pyRD import RD
        from pyRD.core.RDconstant import RDStateDone
        self._rd = RD()
        try:
            self._rd.DeviceEnumLists()
            idx = None
            for i, device in enumerate(self._rd.devicelist):
                sn = device[1].decode() if isinstance(device[1], bytes) else str(device[1])
                desc = device[0].decode() if isinstance(device[0], bytes) else str(device[0])
                if 'YZ' in sn or 'YZ' in desc:
                    idx = i
                    break
            if idx is None:
                raise RuntimeError("未检测到 Raindrop 设备，请检查 USB 连接")
            sts = self._rd.DeviceOpen(idx)
            if sts != 0:
                raise RuntimeError(f"设备打开失败，错误码: {sts}")
            self._connected = True
            return True
        except Exception:
            try:
                self._rd.DeviceClose()
            except Exception:
                pass
            self._rd = None
            raise

    def disconnect(self) -> None:
        if self._rd is not None:
            if self._connected:
                try:
                    self._rd.AnalogOutConfigure(0, False)
                    self._rd.AnalogOutConfigure(1, False)
                    self._rd.AnalogIOChannelEnableSet(0, False)
                    self._rd.AnalogIOChannelEnableSet(1, False)
                except Exception:
                    pass
            try:
                self._rd.DeviceClose()
            except Exception:
                pass
            self._rd = None
            self._connected = False

    def set_voltage(self, channel: int, voltage: float) -> None:
        from pyRD.core.RDconstant import RDFUNCDC, RDAnalogOutNodeCarrier
        voltage = max(0.0, min(voltage, 5.0))
        rd = self._rd
        rd.AnalogOutNodeEnableSet(channel, RDAnalogOutNodeCarrier, True)
        rd.AnalogOutNodeFunctionSet(channel, RDAnalogOutNodeCarrier, RDFUNCDC)
        rd.AnalogOutNodeOffsetAmpSet(channel, RDAnalogOutNodeCarrier, voltage, 0)
        rd.AnalogOutConfigure(channel, True)

    def read_voltage(self, channel: int, samples: int = 2048) -> List[float]:
        from pyRD.core.RDconstant import RDTRIGSRCNone, RDStateDone
        rd = self._rd
        rd.AnalogInCHEnable(channel, True)
        rd.AnalogInCHRangeSet(channel, 5)
        rd.AnalogInFrequencySet(1000000)
        rd.AnalogInBufferSizeSet(samples)
        rd.AnalogInTriggerSourceSet(RDTRIGSRCNone)
        rd.AnalogInRun(True)
        t0 = time.time()
        while time.time() - t0 < ADC_TIMEOUT_S:
            rd.AnalogInStatus()
            if rd.analoginstatus == RDStateDone:
                break
            time.sleep(ADC_POLL_INTERVAL)
        else:
            rd.AnalogInRun(False)
            raise RuntimeError("ADC 采集超时")
        rd.AnalogInRead(samples, channel)
        rd.AnalogInRun(False)
        if channel == 0:
            data = list(rd.aidatach1)
        else:
            data = list(rd.aidatach2)
        if self._save_raw:
            import numpy as np
            import os as _os
            import pathlib as _pl
            d = str(_pl.Path(__file__).parent.parent / "raw_data" / "_driver_dump")
            _os.makedirs(d, exist_ok=True)
            self._raw_counter += 1
            np.save(_os.path.join(d, f"ch{channel}_{self._raw_counter:04d}.npy"),
                    np.array(data, dtype=np.float64))
        return data

    def read_dual_voltage(self, ch0: int, ch1: int, samples: int = 2048) -> tuple:
        from pyRD.core.RDconstant import RDTRIGSRCNone, RDStateDone
        rd = self._rd
        rd.AnalogInCHEnable(ch0, True)
        rd.AnalogInCHRangeSet(ch0, 5)
        rd.AnalogInCHEnable(ch1, True)
        rd.AnalogInCHRangeSet(ch1, 5)
        rd.AnalogInFrequencySet(1000000)
        rd.AnalogInBufferSizeSet(samples)
        rd.AnalogInTriggerSourceSet(RDTRIGSRCNone)
        rd.AnalogInRun(True)
        t0 = time.time()
        while time.time() - t0 < ADC_TIMEOUT_S:
            rd.AnalogInStatus()
            if rd.analoginstatus == RDStateDone:
                break
            time.sleep(ADC_POLL_INTERVAL)
        else:
            rd.AnalogInRun(False)
            raise RuntimeError("ADC 采集超时")
        rd.AnalogInRead(samples, ch0)
        data_ch0 = list(rd.aidatach1 if ch0 == 0 else rd.aidatach2)
        rd.AnalogInRead(samples, ch1)
        data_ch1 = list(rd.aidatach1 if ch1 == 0 else rd.aidatach2)
        rd.AnalogInRun(False)
        return (data_ch0, data_ch1)

    def set_current_source(self, channel: int, current: float) -> None:
        voltage = current * R_BASE
        self.set_voltage(channel, voltage)

    def read_current(self, channel: int, samples: int = 2048) -> List[float]:
        voltages = self.read_voltage(channel, samples)
        return [v / R_SENSE for v in voltages]

    def set_power_supply(self, channel: int, voltage: float,
                         enable: bool = True) -> None:
        self._rd.AnalogIOChannelNodeSet(channel, voltage)
        self._rd.AnalogIOChannelEnableSet(channel, enable)

    def emergency_stop(self) -> None:
        if self._rd:
            self._rd.AnalogOutConfigure(0, False)
            self._rd.AnalogOutConfigure(1, False)
            self._rd.AnalogInRun(False)
            self._rd.AnalogIOChannelEnableSet(0, False)
            self._rd.AnalogIOChannelEnableSet(1, False)

    def dmm_read_dc_voltage(self) -> float:
        from pyRD.core.RDconstant import RDDMMDCV
        rd = self._rd
        rd.DMMOpen(True)
        rd.DMMSet(RDDMMDCV, 1)
        import time; time.sleep(0.3)
        rd.RDDMMReadSingle()
        raw = rd.DMMData.value
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8', errors='replace')
        raw = raw.strip().rstrip('\x00')
        rd.DMMOpen(False)
        return float(raw.rstrip('V').strip())

    def dmm_read_diode(self) -> float:
        from pyRD.core.RDconstant import RDDMMDiode
        rd = self._rd
        rd.DMMOpen(True)
        rd.DMMSet(RDDMMDiode, 0)
        import time; time.sleep(0.5)
        rd.RDDMMReadSingle()
        raw = rd.DMMData.value
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8', errors='replace')
        raw = raw.strip().rstrip('\x00')
        rd.DMMOpen(False)
        return float(raw.rstrip('V').strip())

    def set_digital_output(self, channel: int, state: bool) -> None:
        rd = self._rd
        rd.DigitalIOOutputEnableSet(1 << channel)
        current = rd.DigitalIOInputStatus() if hasattr(rd, 'DigitalIOInputStatus') else 0
        if state:
            rd.DigitalIOOutputSet(current | (1 << channel))
        else:
            rd.DigitalIOOutputSet(current & ~(1 << channel))

    def read_digital_input(self) -> int:
        return self._rd.DigitalIOInputStatus()


def get_sdk_version() -> str:
    """读取 IP-SDK 版本号"""
    try:
        from ctypes import create_string_buffer, byref
        from pyRD import RD
        rd = RD()
        return rd.sdk_version if hasattr(rd, 'sdk_version') else "unknown"
    except Exception:
        return "unknown"


def detect_device() -> tuple:
    """检测设备连接状态，返回 (found: bool, info: dict)
    info 字典包含: serial, model, sdk_version, message
    """
    result = {"serial": "", "model": "", "sdk_version": "", "message": ""}
    try:
        from ctypes import create_string_buffer, byref
        from pyRD import RD
        rd = RD()

        # 读取 SDK 版本
        ver_buf = create_string_buffer(64)
        rd.dll.RDSDKVersion(byref(ver_buf))
        result["sdk_version"] = ver_buf.value.decode('utf-8', errors='replace').strip()

        rd.DeviceEnumLists()
        for device in rd.devicelist:
            sn = device[1].decode() if isinstance(device[1], bytes) else str(device[1])
            desc = device[0].decode() if isinstance(device[0], bytes) else str(device[0])
            if 'YZ' in sn or 'YZ' in desc:
                result["serial"] = sn
                result["model"] = desc
                result["message"] = f"SN:{sn} ({desc})"
                return True, result
        if rd.devicelist:
            all_devs = "; ".join(
                f"{d[1].decode() if isinstance(d[1], bytes) else d[1]}"
                for d in rd.devicelist)
            result["message"] = f"发现设备但非YZ系列: {all_devs}"
            return False, result
        result["message"] = "未发现任何USB设备"
        return False, result
    except Exception as e:
        result["message"] = f"SDK异常: {e}"
        return False, result
