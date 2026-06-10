"""2N3904 BJT 五项核心参数测试函数"""
import time
import numpy as np
import sys
sys.path.append(r'"C:\Users\lenovo\Desktop\jichuangsai-report\IPSDK3.2\IP-SDK\Python\src"')
from pyRD.core.RDconstant import *
from config import *
from data_process import trimmed_mean
from safety import safety_check, emergency_reset


def connect_device():
    """枚举设备、打开连接、读取校准数据"""
    from pyRD import RD
    rd = RD()
    rd.DeviceEnumLists()
    usb_port = None
    for i, dev in enumerate(rd.devicelist):
        if b'YZ' in dev[1]:
            usb_port = i
            break
    if usb_port is None:
        raise RuntimeError("未找到 Model3 设备")
    status = rd.DeviceOpen(usb_port)
    if status != 0:
        raise RuntimeError(f"设备打开失败，错误码: {status}")
    return rd


def _adc_single_read(rd, buffer=BUFFER):
    """执行一次 ADC 采集并等待完成"""
    rd.AnalogInRun(True)
    for _ in range(10):
        rd.AnalogInStatus()
        if rd.analoginstatus == RDStateDone:
            break
        time.sleep(0.01)


def test_hFE(rd, ib_list=None, vce_set=5.0):
    """
    直流电流增益测试
    固定 VCE，扫描 IB，测量 IC，计算 hFE = IC / IB
    """
    if ib_list is None:
        ib_list = [1e-6, 2e-6, 5e-6, 10e-6, 20e-6, 50e-6]
    results = []
    # 配置 AWG CH1 输出固定 VCE
    rd.AnalogOutNodeEnableSet(1, RDAnalogOutNodeCarrier, True)
    rd.AnalogOutNodeFunctionSet(1, RDAnalogOutNodeCarrier, RDFUNCDC)
    rd.AnalogOutNodeOffsetAmpSet(1, RDAnalogOutNodeCarrier, vce_set, 0)
    rd.AnalogOutConfigure(1, True)
    # 配置 ADC
    rd.AnalogInCHEnable(0, True); rd.AnalogInCHRangeSet(0, 5)
    rd.AnalogInCHEnable(1, True); rd.AnalogInCHRangeSet(1, 5)
    rd.AnalogInFrequencySet(FS_RATE)
    rd.AnalogInTriggerSourceSet(RDTRIGSRCNone)
    rd.AnalogInBufferSizeSet(BUFFER)

    for ib_target in ib_list:
        vb_set = ib_target * R_BASE + 0.7
        rd.AnalogOutNodeEnableSet(0, RDAnalogOutNodeCarrier, True)
        rd.AnalogOutNodeFunctionSet(0, RDAnalogOutNodeCarrier, RDFUNCDC)
        rd.AnalogOutNodeOffsetAmpSet(0, RDAnalogOutNodeCarrier, vb_set, 0)
        rd.AnalogOutConfigure(0, True)
        time.sleep(0.1)
        _adc_single_read(rd)
        rd.AnalogInRead(BUFFER, 0)
        rd.AnalogInRead(BUFFER, 1)
        vbe = trimmed_mean(list(rd.aidatach1), TRIM_PCT)
        vce_meas = trimmed_mean(list(rd.aidatach2), TRIM_PCT)
        ib_actual = (vb_set - vbe) / R_BASE
        ic = (vce_set - vce_meas) / R_SENSE
        safety_check(rd, ic, vce_meas)
        hfe = ic / ib_actual if ib_actual > 0 else 0
        results.append((ib_actual, ic, hfe))

    rd.AnalogOutConfigure(0, False)
    rd.AnalogOutConfigure(1, False)
    rd.AnalogInRun(False)
    return results


def test_vce_sat(rd, ic_set=10e-3, ib_set=1e-3):
    """饱和压降测试（开尔文四线制）"""
    vb = ib_set * R_BASE + 0.7
    rd.AnalogOutNodeEnableSet(0, RDAnalogOutNodeCarrier, True)
    rd.AnalogOutNodeFunctionSet(0, RDAnalogOutNodeCarrier, RDFUNCDC)
    rd.AnalogOutNodeOffsetAmpSet(0, RDAnalogOutNodeCarrier, vb, 0)
    rd.AnalogOutConfigure(0, True)
    vce_force = ic_set * R_SENSE + 0.3
    rd.AnalogOutNodeEnableSet(1, RDAnalogOutNodeCarrier, True)
    rd.AnalogOutNodeFunctionSet(1, RDAnalogOutNodeCarrier, RDFUNCDC)
    rd.AnalogOutNodeOffsetAmpSet(1, RDAnalogOutNodeCarrier, vce_force, 0)
    rd.AnalogOutConfigure(1, True)
    time.sleep(0.2)
    rd.AnalogInCHEnable(0, True); rd.AnalogInCHRangeSet(0, 5)
    rd.AnalogInCHEnable(1, True); rd.AnalogInCHRangeSet(1, 5)
    rd.AnalogInFrequencySet(FS_RATE)
    rd.AnalogInTriggerSourceSet(RDTRIGSRCNone)
    rd.AnalogInBufferSizeSet(BUFFER)
    _adc_single_read(rd)
    rd.AnalogInRead(BUFFER, 0)
    rd.AnalogInRead(BUFFER, 1)
    v_sense_p = trimmed_mean(list(rd.aidatach1), TRIM_PCT)
    v_sense_n = trimmed_mean(list(rd.aidatach2), TRIM_PCT)
    vce_sat = v_sense_p - v_sense_n
    rd.AnalogOutConfigure(0, False)
    rd.AnalogOutConfigure(1, False)
    rd.AnalogInRun(False)
    return vce_sat


def test_icbo(rd, vcb_set=5.0):
    """集电极-基极反向截止电流测试"""
    rd.AnalogOutNodeEnableSet(1, RDAnalogOutNodeCarrier, True)
    rd.AnalogOutNodeFunctionSet(1, RDAnalogOutNodeCarrier, RDFUNCDC)
    rd.AnalogOutNodeOffsetAmpSet(1, RDAnalogOutNodeCarrier, vcb_set, 0)
    rd.AnalogOutConfigure(1, True)
    rd.AnalogOutNodeEnableSet(0, RDAnalogOutNodeCarrier, True)
    rd.AnalogOutNodeFunctionSet(0, RDAnalogOutNodeCarrier, RDFUNCDC)
    rd.AnalogOutNodeOffsetAmpSet(0, RDAnalogOutNodeCarrier, 0, 0)
    rd.AnalogOutConfigure(0, True)
    time.sleep(0.5)
    rd.AnalogInCHEnable(0, True); rd.AnalogInCHRangeSet(0, 5)
    rd.AnalogInFrequencySet(FS_RATE)
    rd.AnalogInTriggerSourceSet(RDTRIGSRCNone)
    rd.AnalogInBufferSizeSet(BUFFER)
    _adc_single_read(rd)
    rd.AnalogInRead(BUFFER, 0)
    v_out = trimmed_mean(list(rd.aidatach1), TRIM_PCT)
    icbo = v_out / R_LEAK
    rd.AnalogOutConfigure(0, False)
    rd.AnalogOutConfigure(1, False)
    rd.AnalogInRun(False)
    return icbo


def test_iceo(rd, vce_set=5.0):
    """集电极-发射极反向截止电流测试"""
    rd.AnalogOutNodeEnableSet(1, RDAnalogOutNodeCarrier, True)
    rd.AnalogOutNodeFunctionSet(1, RDAnalogOutNodeCarrier, RDFUNCDC)
    rd.AnalogOutNodeOffsetAmpSet(1, RDAnalogOutNodeCarrier, vce_set, 0)
    rd.AnalogOutConfigure(1, True)
    rd.AnalogOutConfigure(0, False)  # 基极悬空
    time.sleep(0.5)
    rd.AnalogInCHEnable(0, True); rd.AnalogInCHRangeSet(0, 5)
    rd.AnalogInFrequencySet(FS_RATE)
    rd.AnalogInTriggerSourceSet(RDTRIGSRCNone)
    rd.AnalogInBufferSizeSet(BUFFER)
    _adc_single_read(rd)
    rd.AnalogInRead(BUFFER, 0)
    v_out = trimmed_mean(list(rd.aidatach1), TRIM_PCT)
    iceo = v_out / R_LEAK
    rd.AnalogOutConfigure(1, False)
    rd.AnalogInRun(False)
    return iceo


def test_bvceo(rd, ic_threshold=100e-6):
    """低压漏电流趋势测试（原 BVCEO 击穿测试，改为 5V 内安全扫描）"""
    trend = []
    rd.AnalogOutConfigure(0, False)
    rd.AnalogInCHEnable(0, True); rd.AnalogInCHRangeSet(0, 5)
    rd.AnalogInFrequencySet(FS_RATE)
    rd.AnalogInTriggerSourceSet(RDTRIGSRCNone)
    rd.AnalogInBufferSizeSet(BUFFER)
    for vce_step in np.arange(1.0, 5.5, 0.5):
        rd.AnalogOutNodeEnableSet(1, RDAnalogOutNodeCarrier, True)
        rd.AnalogOutNodeFunctionSet(1, RDAnalogOutNodeCarrier, RDFUNCDC)
        rd.AnalogOutNodeOffsetAmpSet(1, RDAnalogOutNodeCarrier, vce_step, 0)
        rd.AnalogOutConfigure(1, True)
        time.sleep(0.3)
        _adc_single_read(rd)
        rd.AnalogInRead(BUFFER, 0)
        v_leak = trimmed_mean(list(rd.aidatach1), TRIM_PCT)
        ic_meas = v_leak / R_LEAK
        trend.append({"vce": vce_step, "ic_leak": ic_meas})
    rd.AnalogOutConfigure(1, False)
    rd.AnalogInRun(False)
    return trend
