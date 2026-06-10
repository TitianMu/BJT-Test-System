"""
datasheet_curves.py - 2N3904 数据手册特性曲线生成
基于 Gummel-Poon 模型或实测扫描数据生成 6 张学术出版级特性曲线。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.ndimage import gaussian_filter1d

from bjt_model import GummelPoonBJT, KEC_2N3904, PRESETS
from data_processor import savgol_smooth

# ===== 学术样式 =====
ACADEMIC_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
    'xtick.top': True,
    'ytick.right': True,
    'legend.fontsize': 8,
    'legend.framealpha': 0.9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
}


def _setup_log_grid(ax, which_axes='both'):
    """为对数坐标添加主/次网格线。"""
    ax.grid(True, which='major', linewidth=0.5, alpha=0.4, color='gray')
    ax.grid(True, which='minor', linewidth=0.3, alpha=0.2, color='gray')
    if 'x' in which_axes or which_axes == 'both':
        ax.xaxis.set_minor_locator(LogLocator(subs='auto', numticks=20))
    if 'y' in which_axes or which_axes == 'both':
        ax.yaxis.set_minor_locator(LogLocator(subs='auto', numticks=20))


def _setup_linear_grid(ax):
    """为线性坐标添加网格线。"""
    ax.grid(True, which='major', linewidth=0.5, alpha=0.4, color='gray')
    ax.minorticks_on()
    ax.grid(True, which='minor', linewidth=0.3, alpha=0.2, color='gray')


def _smooth_measured(data, window=7, polyorder=2):
    """Savitzky-Golay 平滑：保持峰值位置和边缘值。
    委托给 data_processor.savgol_smooth 统一实现。"""
    return savgol_smooth(data, window=window, polyorder=polyorder)


# ==================== 6 张曲线 ====================
def plot_ic_vs_vce(model, ax, measured_data=None):
    """曲线1: IC - VCE 输出特性族 (Ta=25C)"""
    if measured_data:
        vce = np.array(measured_data["vce_array"])
        for i, ib in enumerate(measured_data["ib_values"]):
            ic_ma = _smooth_measured(np.array(measured_data["ic_matrix"][i]) * 1e3)
            ax.plot(vce, ic_ma, '-', linewidth=0.8)
            label_y = ic_ma[-1]
            if label_y > 0.1:
                ax.annotate(f'{ib*1e6:.0f}',
                            xy=(vce[-1], label_y),
                            fontsize=6, ha='left', va='center',
                            xytext=(3, 0), textcoords='offset points')
    else:
        vce = np.concatenate([np.linspace(0.005, 0.3, 80),
                              np.linspace(0.3, 10.0, 200)])
        ib_values = [10e-6, 20e-6, 50e-6, 100e-6, 150e-6,
                     200e-6, 250e-6, 300e-6, 400e-6, 500e-6]
        for ib in ib_values:
            ic = model.ic_vce_curve(ib, vce)
            ic_ma = ic * 1e3
            ax.plot(vce, ic_ma, 'k-', linewidth=0.8)
            label_y = ic_ma[-1]
            if label_y > 0.1:
                ax.annotate(f'{ib*1e6:.0f}',
                            xy=(vce[-1], label_y),
                            fontsize=6, ha='left', va='center',
                            xytext=(3, 0), textcoords='offset points')
    ax.set_xlabel(r'COLLECTOR-EMITTER VOLTAGE  $V_{CE}$  (V)')
    ax.set_ylabel(r'COLLECTOR CURRENT  $I_C$  (mA)')
    ax.set_title(r'$I_C$  -  $V_{CE}$')
    ax.set_xlim(0, 5 if measured_data else 10)
    ax.set_ylim(0, 60)
    ax.text(0.05, 0.95, 'COMMON EMITTER\nTa=25°C',
            transform=ax.transAxes, fontsize=7, va='top')
    ax.text(0.95, 0.05, r'$I_B$=10~500$\mu$A',
            transform=ax.transAxes, fontsize=7, ha='right')
    _setup_linear_grid(ax)


def plot_hfe_vs_ic(model, ax, measured_data=None):
    """曲线2: hFE - IC 直流增益特性 (Ta=25C)"""
    if measured_data:
        ic_raw = np.array(measured_data["ic_array"])
        hfe_raw = np.array(measured_data["hfe_array"])
        mask = ic_raw > 1e-6
        ic_ma = ic_raw[mask] * 1e3
        n = len(hfe_raw[mask])
        win = min(15, n if n % 2 == 1 else n - 1)
        win = max(win, 5)
        hfe = _smooth_measured(hfe_raw[mask], window=win, polyorder=2)
        ax.loglog(ic_ma, hfe, '-', linewidth=1.2, color='#1a3a5c')
    else:
        ic_array = np.logspace(-5, -0.3, 300)
        hfe = model.hfe_vs_ic(ic_array, vce=1.0)
        hfe = gaussian_filter1d(hfe, sigma=2)
        ic_ma = ic_array * 1e3
        ax.loglog(ic_ma, hfe, 'k-', linewidth=1.2)
    ax.set_xlabel(r'COLLECTOR CURRENT  $I_C$  (mA)')
    ax.set_ylabel(r'DC CURRENT GAIN  $h_{FE}$')
    ax.set_title(r'$h_{FE}$  -  $I_C$')
    ax.set_xlim(1, 100)
    ax.set_ylim(10, 1000)
    vce_label = '5V' if measured_data else '1V'
    ax.text(0.95, 0.95, f'COMMON EMITTER\n$V_{{CE}}$={vce_label}\nTa=25°C',
            transform=ax.transAxes, fontsize=7, va='top', ha='right')
    _setup_log_grid(ax)


def plot_vbe_sat_vs_ic(model, ax, measured_data=None):
    """曲线3: VBE(sat) - IC 基极饱和压降 (Ta=25C)"""
    if measured_data:
        ic_ma = np.array(measured_data["ic_array"]) * 1e3
        vbe_sat_mv = _smooth_measured(np.array(measured_data["vbe_sat_array"]) * 1e3)
        ax.semilogx(ic_ma, vbe_sat_mv, '-', linewidth=1.2, color='#1a3a5c')
    else:
        ic_array = np.logspace(-5, -0.3, 200)
        vbe_sat = model.vbe_sat_vs_ic(ic_array, ratio=10.0)
        vbe_sat = gaussian_filter1d(vbe_sat, sigma=2)
        ic_ma = ic_array * 1e3
        ax.semilogx(ic_ma, vbe_sat * 1e3, 'k-', linewidth=1.2)
    ax.set_xlabel(r'COLLECTOR CURRENT  $I_C$  (mA)')
    ax.set_ylabel(r'BASE-EMITTER SATURATION' + '\n' +
                  r'VOLTAGE  $V_{BE(sat)}$  (mV)')
    ax.set_title(r'$V_{BE(sat)}$  -  $I_C$')
    ax.set_xlim(0.01, 500)
    ax.set_ylim(500, 900)
    ax.text(0.05, 0.95, 'COMMON EMITTER\n$I_C/I_B$=10\nTa=25°C',
            transform=ax.transAxes, fontsize=7, va='top')
    _setup_log_grid(ax, which_axes='x')
    ax.minorticks_on()
    ax.grid(True, which='minor', linewidth=0.3, alpha=0.2, color='gray',
            axis='x')


def plot_vce_sat_vs_ic(model, ax, measured_data=None):
    """曲线4: VCE(sat) - IC 集电极饱和压降 (Ta=25C)"""
    if measured_data:
        ic_ma = np.array(measured_data["ic_array"]) * 1e3
        vce_sat_mv = _smooth_measured(np.array(measured_data["vce_sat_array"]) * 1e3)
        ax.loglog(ic_ma, vce_sat_mv, '-', linewidth=1.2, color='#1a3a5c')
    else:
        ic_array = np.logspace(-5, -0.3, 200)
        vce_sat = model.vce_sat_vs_ic(ic_array, ratio=10.0)
        vce_sat = gaussian_filter1d(vce_sat, sigma=2)
        ic_ma = ic_array * 1e3
        ax.loglog(ic_ma, vce_sat * 1e3, 'k-', linewidth=1.2)
    ax.set_xlabel(r'COLLECTOR CURRENT  $I_C$  (mA)')
    ax.set_ylabel(r'COLLECTOR-EMITTER SATURATION' + '\n' +
                  r'VOLTAGE  $V_{CE(sat)}$  (mV)')
    ax.set_title(r'$V_{CE(sat)}$  -  $I_C$')
    ax.set_xlim(0.01, 500)
    ax.set_ylim(1, 1000)
    ax.text(0.05, 0.95, 'COMMON EMITTER\n$I_C/I_B$=10\nTa=25°C',
            transform=ax.transAxes, fontsize=7, va='top')
    _setup_log_grid(ax)


def plot_ic_vs_vbe(model, ax, measured_data=None):
    """曲线5: IC - VBE 转移特性 (Ta=25C)"""
    if measured_data:
        vbe = np.array(measured_data["vbe_array"])
        ic_ma = _smooth_measured(np.array(measured_data["ic_array"]) * 1e3)
        ax.plot(vbe, ic_ma, '-', linewidth=1.2, color='#1a3a5c')
    else:
        vbe_array = np.linspace(0.3, 1.0, 300)
        ic = model.ic_vs_vbe(vbe_array, vce=5.0)
        ic_ma = ic * 1e3
        ax.plot(vbe_array, ic_ma, 'k-', linewidth=1.2)
    ax.set_xlabel(r'BASE-EMITTER VOLTAGE  $V_{BE}$  (V)')
    ax.set_ylabel(r'COLLECTOR CURRENT  $I_C$  (mA)')
    ax.set_title(r'$I_C$  -  $V_{BE}$')
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0, 75)
    ax.text(0.05, 0.95, 'COMMON EMITTER\n$V_{CE}$=5V\nTa=25°C',
            transform=ax.transAxes, fontsize=7, va='top')
    _setup_linear_grid(ax)


def plot_vce_vs_ib(model, ax, measured_data=None):
    """曲线6: VCE - IB 特性 (Ta=25C)"""
    if measured_data:
        ib_ma = np.array(measured_data["ib_array"]) * 1e3
        for i, ic_t in enumerate(measured_data["ic_targets"]):
            vce = _smooth_measured(np.array(measured_data["vce_matrix"][i]))
            label = f'{ic_t*1e3:.0f}mA'
            ax.semilogx(ib_ma, vce, '-', linewidth=0.8)
            mid = len(vce) // 3
            ax.annotate(f'$I_C$={label}',
                        xy=(ib_ma[mid], vce[mid]),
                        fontsize=6, ha='center', va='bottom',
                        xytext=(0, 4), textcoords='offset points')
    else:
        ib_array = np.logspace(-6, -2, 150)
        ic_targets = [1e-3, 10e-3, 30e-3, 100e-3]
        labels = ['1mA', '10mA', '30mA', '100mA']
        for ic_t, label in zip(ic_targets, labels):
            vce = model.vce_vs_ib(ib_array, ic_t)
            vce = gaussian_filter1d(vce, sigma=3)
            ib_ma = ib_array * 1e3
            ax.semilogx(ib_ma, vce, 'k-', linewidth=0.8)
            mid = np.searchsorted(vce[::-1], 0.5 * vce.max())
            mid = max(len(vce) - mid - 1, len(vce) // 3)
            ax.annotate(f'$I_C$={label}',
                        xy=(ib_ma[mid], vce[mid]),
                        fontsize=6, ha='center', va='bottom',
                        xytext=(0, 4), textcoords='offset points')
    ax.set_xlabel(r'BASE CURRENT  $I_B$  (mA)')
    ax.set_ylabel(r'COLLECTOR-EMITTER VOLTAGE  $V_{CE}$  (V)')
    ax.set_title(r'$V_{CE}$  -  $I_B$')
    ax.set_xlim(0.001, 10)
    ax.set_ylim(0, 5.0)
    ax.text(0.05, 0.05, 'COMMON EMITTER\nTa=25°C',
            transform=ax.transAxes, fontsize=7, va='bottom')
    _setup_log_grid(ax, which_axes='x')
    ax.minorticks_on()


# ==================== 入口函数 ====================

def generate_datasheet_figure(params=None, filename='2N3904_Curves.png',
                              measured_data=None, device_name=None):
    """生成 3x2 网格的 6 张特性曲线。
    params: SPICE 参数字典，None 则使用 KEC-2N3904 默认值。
    measured_data: 实测扫描数据字典（来自 TestEngine.run_all_sweeps），
                   非 None 时使用实测数据绘图，忽略 params。
    返回保存的文件路径。
    """
    model = GummelPoonBJT(params or KEC_2N3904) if not measured_data else None
    with plt.rc_context(ACADEMIC_STYLE):
        fig, axes = plt.subplots(3, 2, figsize=(11, 14))
        plot_ic_vs_vce(model, axes[0, 0],
                       measured_data.get("ic_vs_vce") if measured_data else None)
        plot_hfe_vs_ic(model, axes[0, 1],
                       measured_data.get("hfe_vs_ic") if measured_data else None)
        plot_vbe_sat_vs_ic(model, axes[1, 0],
                           measured_data.get("vbe_sat_vs_ic") if measured_data else None)
        plot_vce_sat_vs_ic(model, axes[1, 1],
                           measured_data.get("vce_sat_vs_ic") if measured_data else None)
        plot_ic_vs_vbe(model, axes[2, 0],
                       measured_data.get("ic_vs_vbe") if measured_data else None)
        plot_vce_vs_ib(model, axes[2, 1],
                       measured_data.get("vce_vs_ib") if measured_data else None)
        if measured_data:
            name = device_name or 'Unknown'
            title = f'{name}  NPN Transistor Characteristics (Measured)'
        else:
            name = device_name or (params.get('name', 'Custom') if params else 'KEC-2N3904')
            title = f'{name}  NPN Transistor Characteristics'
        fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.98], h_pad=3.0, w_pad=2.5)
        fig.savefig(filename, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
    return filename


if __name__ == '__main__':
    import os
    for name, params in PRESETS.items():
        fn = f'{name}_Curves.png'
        generate_datasheet_figure(params, fn)
        print(f'Generated: {os.path.abspath(fn)}')
