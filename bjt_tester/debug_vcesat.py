"""
debug_vcesat.py - 饱和压降诊断脚本
直接连接 Model-3 硬件，采集原始数据，打印诊断信息。
用法：python debug_vcesat.py
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from real_driver import RealInstrumentDriver

R_BASE_SAT = 1e3
R_SENSE = 100.0
BUFFER_SIZE = 2048
IC_SET = 10e-3
IB_SET = 1e-3

CH_VB = 0
CH_VCE = 1


def run_diagnostic():
    drv = RealInstrumentDriver()
    print("正在连接 Model-3...")
    drv.connect()
    print("连接成功\n")

    vb = IB_SET * R_BASE_SAT + 0.7
    vce_force = IC_SET * R_SENSE + 0.3

    print(f"=== 测试参数 ===")
    print(f"IB_set = {IB_SET*1000:.1f} mA, IC_set = {IC_SET*1000:.1f} mA")
    print(f"VB (AWG CH0) = {vb:.3f} V")
    print(f"VCE_force (AWG CH1) = {vce_force:.3f} V")
    print(f"R_BASE_SAT = {R_BASE_SAT:.0f} Ω, R_SENSE = {R_SENSE:.0f} Ω")
    print(f"ADC 采样: {BUFFER_SIZE} samples @ 1MHz\n")

    try:
        drv.set_voltage(CH_VB, vb)
        drv.set_voltage(CH_VCE, vce_force)
        print("AWG 输出已设置，等待 500ms 建立...")
        time.sleep(0.5)

        print("\n=== 连续 10 次采集 ===\n")
        print(f"{'#':<3} {'sp(V)':<10} {'sn(V)':<10} {'sp-sn(mV)':<12} "
              f"{'sn-sp(mV)':<12} {'std_sp(mV)':<12} {'std_sn(mV)':<12}")
        print("-" * 75)

        all_sp = []
        all_sn = []

        for i in range(10):
            raw_sp, raw_sn = drv.read_dual_voltage(0, 1, BUFFER_SIZE)
            sp = np.mean(raw_sp)
            sn = np.mean(raw_sn)
            std_sp = np.std(raw_sp) * 1000
            std_sn = np.std(raw_sn) * 1000
            diff_sp_sn = (sp - sn) * 1000
            diff_sn_sp = (sn - sp) * 1000

            all_sp.append(sp)
            all_sn.append(sn)

            print(f"{i+1:<3} {sp:<10.4f} {sn:<10.4f} {diff_sp_sn:<12.2f} "
                  f"{diff_sn_sp:<12.2f} {std_sp:<12.2f} {std_sn:<12.2f}")
            time.sleep(0.05)

        print("\n=== 汇总分析 ===\n")
        sp_arr = np.array(all_sp)
        sn_arr = np.array(all_sn)
        diff_arr = (sp_arr - sn_arr) * 1000

        print(f"sp 均值: {sp_arr.mean():.4f} V, 标准差: {sp_arr.std()*1000:.2f} mV")
        print(f"sn 均值: {sn_arr.mean():.4f} V, 标准差: {sn_arr.std()*1000:.2f} mV")
        print(f"(sp-sn) 均值: {diff_arr.mean():.2f} mV, 标准差: {diff_arr.std():.2f} mV")
        print(f"(sp-sn) 范围: [{diff_arr.min():.2f}, {diff_arr.max():.2f}] mV")

        print("\n=== 诊断结论 ===\n")

        if sp_arr.mean() > sn_arr.mean():
            print("[极性] sp > sn → VCE(sat) = sp - sn 是正确方向")
            print("       当前代码用 sn - sp 会得到负值 → 极性 BUG 确认！")
        else:
            print("[极性] sn > sp → 当前代码 sn - sp 方向正确")
            print("       或者 sense 线物理接反了")

        avg_std = (np.mean([np.std(raw_sp), np.std(raw_sn)])) * 1000
        if avg_std > 10:
            print(f"[振荡] 单次采集内标准差 > 10mV → 可能存在寄生振荡")
            print("       建议：基极串 47Ω + C-E 并 100pF")
        elif avg_std > 3:
            print(f"[噪声] 标准差 {avg_std:.1f}mV，中等噪声，多次平均可改善")
        else:
            print(f"[噪声] 标准差 {avg_std:.1f}mV，噪声水平正常")

        if diff_arr.std() > 50:
            print(f"[稳定性] 10次测量间波动 > 50mV → 严重不稳定")
            print("         最可能原因：寄生振荡或接触不良")
        elif diff_arr.std() > 10:
            print(f"[稳定性] 10次测量间波动 {diff_arr.std():.1f}mV → 中等不稳定")
        else:
            print(f"[稳定性] 10次测量间波动 {diff_arr.std():.1f}mV → 稳定")

    finally:
        drv.set_voltage(CH_VB, 0)
        drv.set_voltage(CH_VCE, 0)
        drv.disconnect()
        print("\n已关闭输出并断开连接。")


if __name__ == "__main__":
    run_diagnostic()
