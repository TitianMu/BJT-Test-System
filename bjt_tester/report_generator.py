"""
report_generator.py - 测试报告生成模块
生成 Excel 测试报告、IC-VCE 特性曲线 PNG 和数据手册特性曲线。
"""
import os
from datetime import datetime
from typing import Dict, List
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

from datasheet_curves import generate_datasheet_figure

# matplotlib 中文支持
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


def export_excel(results: Dict, filename: str = "BJT_Report.xlsx",
                 driver_mode: str = "virtual",
                 device_name: str = "2N3904") -> str:
    """
    生成 Excel 测试报告。
    参数:
        results:     TestEngine.run_all_tests() 返回的结果字典
        filename:    输出文件名
        driver_mode: 驱动模式标识
    返回:
        保存的文件路径
    """
    if not HAS_OPENPYXL:
        print("openpyxl 未安装，跳过 Excel 报告生成")
        return ""

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = f"{device_name} 测试报告"

    # 报告头
    ws.append([f"{device_name} BJT 全参数自动化测试报告"])
    ws.append(["测试时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    ws.append(["驱动模式", driver_mode])
    ws.append(["被测器件", f"{device_name} NPN BJT"])
    ws.append([])

    # 结果表
    ws.append(["参数", "测量值", "规格范围", "判定结果"])
    specs = {
        "hFE": ("100~630", lambda v: f"{v:.1f}"),
        "VCE(sat)": ("<0.4 V", lambda v: f"{v:.3f} V"),
        "VBE(sat)": ("<1.0 V", lambda v: f"{v*1e3:.1f} mV"),
        "ICBO": ("<100 nA", lambda v: f"{v*1e9:.1f} nA"),
        "ICEO": ("<200 nA", lambda v: f"{v*1e9:.1f} nA"),
        "BVCEO": (">40 V", lambda v: f"{v:.1f} V"),
    }
    for param, (spec, fmt) in specs.items():
        if param in results:
            val = results[param]["value"]
            status = results[param]["status"]
            ws.append([param, fmt(val), spec, status])

    # 设置列宽
    ws.column_dimensions['A'].width = 15
    ws.column_dimensions['B'].width = 15
    ws.column_dimensions['C'].width = 15
    ws.column_dimensions['D'].width = 12

    wb.save(filename)
    return os.path.abspath(filename)


def export_characteristic_curves(results: Dict,
                                 filename: str = "BJT_Curves.png",
                                 dpi: int = 300) -> str:
    """
    生成 hFE-IC 特性曲线和 IC-VCE 输出特性族曲线。
    参数:
        results:  包含 hFE detail 数据的结果字典
        filename: 输出 PNG 文件名
        dpi:      图像分辨率
    返回:
        保存的文件路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左图: hFE vs IC
    if "hFE" in results and "detail" in results["hFE"]:
        detail = results["hFE"]["detail"]
        ic_vals = [d["ic"] * 1e3 for d in detail]  # mA
        hfe_vals = [d["hfe"] for d in detail]
        ax1.plot(ic_vals, hfe_vals, 'bo-', linewidth=2, markersize=6)
        ax1.set_xlabel("IC (mA)")
        ax1.set_ylabel("hFE")
        ax1.set_title("hFE - IC 特性曲线")
        ax1.grid(True, alpha=0.3)

    # 右图: IC-VCE 输出特性族 (模拟)
    vce_range = np.linspace(0, 10, 100)
    ib_values = [5e-6, 10e-6, 20e-6, 50e-6, 100e-6]
    for ib in ib_values:
        IS = 6.734e-15
        VT = 0.02585
        BF = 200.0
        vbe = VT * np.log(ib * BF / IS + 1)
        ic = IS * (np.exp(vbe / VT) - 1) * (1 + vce_range / 100) * BF / (BF + 1)
        ic_ma = ic * 1e3
        ax2.plot(vce_range, ic_ma, linewidth=1.5,
                 label=f"IB={ib*1e6:.0f}uA")
    ax2.set_xlabel("VCE (V)")
    ax2.set_ylabel("IC (mA)")
    ax2.set_title("IC - VCE 输出特性族")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return os.path.abspath(filename)


def export_datasheet_curves(params: dict = None,
                            filename: str = "2N3904_Datasheet_Curves.png",
                            measured_data: dict = None,
                            device_name: str = None) -> str:
    """生成 6 张数据手册特性曲线。
    params: SPICE 参数字典，None 使用 KEC-2N3904 默认值。
    measured_data: 实测扫描数据，非 None 时使用实测数据。
    """
    generate_datasheet_figure(params, filename, measured_data=measured_data,
                              device_name=device_name)
    return os.path.abspath(filename)
