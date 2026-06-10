"""
data_processor.py - 数据处理模块
实现三级信号处理链：3σ异常值剔除 → 截尾均值 → S-G滤波（曲线数据）。
"""
import numpy as np
from typing import List, Optional

# ===== 滤波参数 =====
DEFAULT_TRIM_RATIO = 0.05   # 截尾比例: 去除两端各 5%
SIGMA_THRESHOLD = 3.0       # 异常值剔除阈值: 3-sigma


def trimmed_mean(data: List[float], trim_ratio: float = DEFAULT_TRIM_RATIO,
                 use_sigma: bool = False, sigma: float = SIGMA_THRESHOLD) -> float:
    """
    截尾均值滤波：去除两端各 trim_ratio 比例的极值后取平均。
    可选在截尾前先执行 3σ 异常值剔除。

    对 2048 点数据，trim_ratio=0.05 时去除两端各 102 点，
    对中间 1844 点取均值。

    参数:
        data:       原始采样数据列表
        trim_ratio: 截尾比例，默认 0.05 (5%)
        use_sigma:  是否在截尾前执行 3σ 异常值剔除
        sigma:      3σ 阈值倍数，默认 3.0
    返回:
        截尾均值
    """
    arr = np.array(data, dtype=np.float64)

    if use_sigma:
        mu = np.mean(arr)
        std = np.std(arr)
        if std > 1e-15:
            mask = np.abs(arr - mu) < sigma * std
            arr = arr[mask]

    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    cut = int(n * trim_ratio)
    if cut * 2 >= n:
        return float(np.mean(sorted_arr))
    trimmed = sorted_arr[cut : n - cut]
    return float(np.mean(trimmed))


def remove_outliers(data: List[float],
                    sigma: float = SIGMA_THRESHOLD) -> List[float]:
    """
    基于 3-sigma 准则剔除异常值。
    参数:
        data:  原始数据列表
        sigma: 阈值倍数，默认 3.0
    返回:
        剔除异常值后的数据列表
    """
    arr = np.array(data, dtype=np.float64)
    mean = np.mean(arr)
    std = np.std(arr)
    if std < 1e-15:
        return data
    mask = np.abs(arr - mean) < sigma * std
    return list(arr[mask])


def process_adc_samples(raw: List[float],
                         trim_ratio: float = DEFAULT_TRIM_RATIO,
                         use_sigma: bool = True,
                         sigma: float = SIGMA_THRESHOLD,
                         return_cleaned: bool = False):
    """
    对单次 ADC 采集的 2048 点原始数据执行完整三级处理链的前两级：
    3σ 异常值剔除 → 截尾均值。

    参数:
        raw:            原始 ADC 采样数据（2048 点）
        trim_ratio:     截尾比例
        use_sigma:      是否启用 3σ 剔除
        sigma:          3σ 阈值倍数
        return_cleaned: 若为 True，同时返回剔除异常值后的数据列表
    返回:
        若 return_cleaned=False: float (均值)
        若 return_cleaned=True:  (mean: float, cleaned: np.ndarray)
    """
    arr = np.array(raw, dtype=np.float64)

    if use_sigma:
        mu = np.mean(arr)
        std = np.std(arr)
        if std > 1e-15:
            mask = np.abs(arr - mu) < sigma * std
            arr = arr[mask]

    n_after_sigma = len(arr)
    n_outliers = len(raw) - n_after_sigma

    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    cut = int(n * trim_ratio)
    if cut * 2 >= n:
        result = float(np.mean(sorted_arr))
    else:
        trimmed = sorted_arr[cut : n - cut]
        result = float(np.mean(trimmed))

    if return_cleaned:
        return result, arr
    return result


def zero_offset_correction(data: List[float],
                           offset: float = 0.0) -> List[float]:
    """
    零点偏移校正：减去系统零点偏移量。
    参数:
        data:   原始数据列表
        offset: 零点偏移量 (V)，通过短路校准获得
    返回:
        校正后的数据列表
    """
    return [x - offset for x in data]


def savgol_smooth(y_data, window: int = 11, polyorder: int = 3):
    """
    三级处理链第三级：Savitzky-Golay 滤波，用于特性曲线平滑。
    在局部窗口内做多项式最小二乘拟合，保留峰值位置与拐点。

    参数:
        y_data:    离散采样点序列（list 或 np.ndarray）
        window:    滑动窗口长度，必须是奇数，默认 11
        polyorder: 多项式阶数，默认 3
    返回:
        np.ndarray: 平滑后的序列，长度与输入相同
    """
    from scipy.signal import savgol_filter
    arr = np.asarray(y_data, dtype=np.float64)
    if len(arr) < window:
        return arr.copy()
    w = min(window, len(arr))
    if w % 2 == 0:
        w -= 1
    if w <= polyorder:
        return arr.copy()
    return savgol_filter(arr, window_length=w, polyorder=polyorder)


def calculate_hfe(vb_set: float, vbe: float, vce_set: float,
                  vce_meas: float, r_base: float,
                  r_sense: float) -> dict:
    """
    计算 hFE 及相关参数。
    返回: {"ib": float, "ic": float, "hfe": float}
    """
    ib = (vb_set - vbe) / r_base
    ic = (vce_set - vce_meas) / r_sense
    hfe = ic / ib if ib > 1e-12 else 0.0
    return {"ib": ib, "ic": ic, "hfe": hfe}


def calculate_leakage(v_transimpedance: float,
                      r_feedback: float) -> float:
    """
    计算漏电流: I = V / Rf
    参数:
        v_transimpedance: 跨阻放大器输出电压 (V)
        r_feedback:       反馈电阻值 (Ohm)
    返回:
        漏电流值 (A)
    """
    return v_transimpedance / r_feedback
