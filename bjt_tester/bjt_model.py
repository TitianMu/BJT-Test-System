"""
bjt_model.py - Gummel-Poon BJT 物理模型
基于 SPICE Gummel-Poon 模型，支持 KEC-2N3904 和 JSCJ-2N3904 两套参数。
所有曲线由物理方程驱动，参数变化时自动更新。
"""
import os
import json
import numpy as np
import warnings
from scipy.optimize import brentq, fsolve

# ===== 物理常数 =====
K_BOLTZMANN = 1.380649e-23   # J/K
Q_ELECTRON = 1.602176634e-19  # C
T_NOMINAL = 298.15            # 25°C in K

# ===== KEC-2N3904 SPICE 参数 =====
KEC_2N3904 = {
    'name': 'KEC-2N3904',
    'IS':  6.734e-15,   # 传输饱和电流 (A)
    'BF':  500.0,       # 理想最大正向 beta
    'NF':  1.0,         # 正向发射系数
    'VAF': 74.03,       # 正向 Early 电压 (V)
    'IKF': 0.06678,     # BF 高电流拐点 (A)
    'ISE': 5.0e-12,     # B-E 漏电饱和电流 (A)
    'NE':  1.8,         # B-E 漏电发射系数
    'BR':  6.092,      # 理想最大反向 beta
    'NR':  1.0,         # 反向发射系数
    'VAR': 28.0,        # 反向 Early 电压 (V)
    'RB':  10.0,        # 基极电阻 (Ohm)
    'RE':  0.0,         # 发射极电阻 (Ohm)
    'RC':  1.0,         # 集电极电阻 (Ohm)
}

# ===== JSCJ-2N3904 SPICE 参数 =====
JSCJ_2N3904 = {
    'name': 'JSCJ-2N3904',
    'IS':  6.734e-15,
    'BF':  650.0,       # 更高的峰值 hFE（数据手册 max=400）
    'NF':  1.0,
    'VAF': 74.03,
    'IKF': 0.06678,
    'ISE': 5.0e-12,
    'NE':  1.8,
    'BR':  6.092,
    'NR':  1.0,
    'VAR': 28.0,
    'RB':  10.0,
    'RE':  0.0,
    'RC':  1.0,
}

# ===== LGE-2N3904 SPICE 参数（LGE Semiconductor 2N3904） =====
# 数据手册: hFE=100~400@IC=10mA, VCE(sat)=0.3V@IC=50mA/IB=5mA
LGE_2N3904 = {
    'name': 'LGE-2N3904',
    'IS':  6.734e-15,
    'BF':  416.4,
    'NF':  1.0,
    'VAF': 74.03,
    'IKF': 0.06678,
    'ISE': 6.5e-12,
    'NE':  1.8,
    'BR':  6.092,
    'NR':  1.0,
    'VAR': 28.0,
    'RB':  10.0,
    'RE':  0.1,
    'RC':  1.0,
}

# ===== ON-2N3904 SPICE 参数（ON Semiconductor / Fairchild 2N3904） =====
# 数据手册: hFE=100~300@IC=10mA, VCE(sat)<0.2V@IC=10mA/IB=1mA
ON_2N3904 = {
    'name': 'ON-2N3904',
    'IS':  6.734e-15,
    'BF':  300.0,
    'NF':  1.0,
    'VAF': 74.03,
    'IKF': 0.06678,
    'ISE': 4.0e-12,
    'NE':  1.8,
    'BR':  6.092,
    'NR':  1.0,
    'VAR': 28.0,
    'RB':  10.0,
    'RE':  0.0,
    'RC':  1.0,
}

# ===== ON-BC337 SPICE 参数（ON Semiconductor BC337-25） =====
# 数据手册: VCEO=45V, hFE=160~400@IC=100mA, VCE(sat)<0.7V@IC=500mA
ON_BC337 = {
    'name': 'ON-BC337',
    'IS':  1.8e-14,     # 传输饱和电流 (A)
    'BF':  400.0,       # 正向电流增益（数据手册 hFE 160~400）
    'NF':  1.0,         # 正向发射系数
    'VAF': 80.0,        # 正向 Early 电压 (V)
    'IKF': 0.5,         # 高电流拐点 (A)，IC(max)=800mA
    'ISE': 5.0e-13,     # B-E 漏电饱和电流 (A)
    'NE':  1.46,        # B-E 漏电发射系数
    'BR':  35.5,        # 反向电流增益
    'NR':  1.0,         # 反向发射系数
    'VAR': 12.5,        # 反向 Early 电压 (V)
    'RB':  0.56,        # 基极电阻 (Ohm)
    'RE':  0.6,         # 发射极电阻 (Ohm)
    'RC':  0.5,         # 集电极电阻 (Ohm)
}

# ===== JSCJ-BC337 SPICE 参数（江苏长晶 BC337） =====
# 数据手册: VCEO=45V, hFE=100~630@IC=100mA, VCE(sat)<0.7V@IC=500mA
JSCJ_BC337 = {
    'name': 'JSCJ-BC337',
    'IS':  2.0e-14,     # 传输饱和电流 (A)
    'BF':  420.0,       # 正向电流增益（数据手册 hFE 100~630，typ 偏高）
    'NF':  1.0,         # 正向发射系数
    'VAF': 75.0,        # 正向 Early 电压 (V)
    'IKF': 0.45,        # 高电流拐点 (A)
    'ISE': 6.0e-13,     # B-E 漏电饱和电流 (A)
    'NE':  1.5,         # B-E 漏电发射系数
    'BR':  30.0,        # 反向电流增益
    'NR':  1.0,         # 反向发射系数
    'VAR': 14.0,        # 反向 Early 电压 (V)
    'RB':  0.8,         # 基极电阻 (Ohm)
    'RE':  0.5,         # 发射极电阻 (Ohm)
    'RC':  0.8,         # 集电极电阻 (Ohm)
}

# 预设参数列表（方便 GUI 枚举）
PRESETS = {
    'KEC-2N3904': KEC_2N3904,
    'JSCJ-2N3904': JSCJ_2N3904,
    'LGE-2N3904': LGE_2N3904,
    'ON-2N3904': ON_2N3904,
    'ON-BC337': ON_BC337,
    'JSCJ-BC337': JSCJ_BC337,
}

# GP 参数的显示名称、单位、默认值（供 GUI 使用）
PARAM_INFO = [
    ('IS',  '传输饱和电流', 'A',   6.734e-15),
    ('BF',  '正向电流增益', '',    500.0),
    ('NF',  '正向发射系数', '',    1.0),
    ('VAF', '正向Early电压', 'V',  74.03),
    ('IKF', '高电流拐点',   'A',   0.06678),
    ('ISE', 'B-E漏电饱和电流', 'A', 5.0e-12),
    ('NE',  'B-E漏电发射系数', '',  1.8),
    ('BR',  '反向电流增益', '',    6.092),
    ('NR',  '反向发射系数', '',    1.0),
    ('VAR', '反向Early电压', 'V',  28.0),
    ('RB',  '基极电阻',     'Ohm', 10.0),
    ('RE',  '发射极电阻',   'Ohm', 0.0),
    ('RC',  '集电极电阻',   'Ohm', 1.0),
]

# ===== 用户预设持久化 =====
_USER_PRESETS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'user_presets.json')


def load_user_data() -> dict:
    if not os.path.exists(_USER_PRESETS_FILE):
        return {"presets": {}, "deleted_builtins": []}
    try:
        with open(_USER_PRESETS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data.setdefault("presets", {})
        data.setdefault("deleted_builtins", [])
        return data
    except (json.JSONDecodeError, IOError):
        return {"presets": {}, "deleted_builtins": []}


def save_user_data(data: dict):
    with open(_USER_PRESETS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class GummelPoonBJT:
    """Gummel-Poon NPN BJT 模型，25°C 工作点计算。"""

    def __init__(self, params: dict = None):
        p = params or KEC_2N3904
        self.name = p.get('name', 'Custom')
        self.IS  = p['IS']
        self.BF  = p['BF']
        self.NF  = p['NF']
        self.VAF = p['VAF']
        self.IKF = p['IKF']
        self.ISE = p['ISE']
        self.NE  = p['NE']
        self.BR  = p['BR']
        self.NR  = p['NR']
        self.VAR = p['VAR']
        self.RB  = p['RB']
        self.RE  = p['RE']
        self.RC  = p['RC']
        self.VT  = K_BOLTZMANN * T_NOMINAL / Q_ELECTRON  # ~25.85 mV

    # ==================== 核心物理方程 ====================

    def _safe_exp(self, x):
        """防溢出指数函数"""
        return np.exp(np.clip(x, -500, 500))

    def compute_ic(self, vbe, vce):
        """计算集电极电流 IC (A)，含 Early 效应和高电流滚降。"""
        vbc = vbe - vce
        # 正向传输电流
        i_f = self.IS * (self._safe_exp(vbe / (self.NF * self.VT)) - 1)
        # 反向传输电流
        i_r = self.IS * (self._safe_exp(vbc / (self.NR * self.VT)) - 1)
        # 基区电荷归一化因子
        q1_inv = 1.0 - vbc / self.VAF
        if self.VAR > 0:
            q1_inv -= vbe / self.VAR
        q1_inv = max(q1_inv, 0.01)
        q1 = 1.0 / q1_inv
        # 高电流滚降
        q2 = i_f / self.IKF if self.IKF > 0 else 0.0
        qb = q1 * 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * q2))
        ic = i_f / qb - i_r / self.BR
        return max(ic, 0.0)

    def compute_ib(self, vbe, vce):
        """计算基极电流 IB (A)，含低电流复合项。"""
        vbc = vbe - vce
        i_f = self.IS * (self._safe_exp(vbe / (self.NF * self.VT)) - 1)
        i_r = self.IS * (self._safe_exp(vbc / (self.NR * self.VT)) - 1)
        # 理想基极电流
        ib_f = i_f / self.BF
        ib_r = i_r / self.BR
        # 低电流复合（非理想基极电流）
        ib_re = self.ISE * (self._safe_exp(vbe / (self.NE * self.VT)) - 1)
        ib = ib_f + ib_re + ib_r
        return max(ib, 0.0)

    def compute_hfe(self, vbe, vce):
        """计算直流电流增益 hFE = IC / IB。"""
        ic = self.compute_ic(vbe, vce)
        ib = self.compute_ib(vbe, vce)
        if ib < 1e-18:
            return 0.0
        return ic / ib

    # ==================== 求解器 ====================

    def solve_vbe_for_ic(self, ic_target, vce):
        """给定目标 IC 和 VCE，反解 VBE。"""
        def f(vbe):
            return self.compute_ic(vbe, vce) - ic_target
        try:
            vbe = brentq(f, 0.1, 1.2, xtol=1e-12)
        except ValueError:
            vbe = 0.7
        return vbe

    def solve_vbe_for_ib(self, ib_target, vce):
        """给定目标 IB 和 VCE，反解 VBE。"""
        def f(vbe):
            return self.compute_ib(vbe, vce) - ib_target
        try:
            vbe = brentq(f, 0.1, 1.2, xtol=1e-12)
        except ValueError:
            vbe = 0.65
        return vbe

    def solve_saturation(self, ic_target, ib_target):
        """求解饱和区工作点：给定 IC 和 IB，求端子 VBE 和 VCE。
        考虑寄生电阻 RE/RC 对端子电压的影响：
        VBE_terminal = VBE_intrinsic + IE * RE
        VCE_terminal = VCE_intrinsic + IC * RC + IE * RE
        """
        ie = ic_target + ib_target

        def ic_error(vce_int):
            vce_int = max(vce_int, 0.001)
            try:
                vbe_int = self.solve_vbe_for_ib(ib_target, vce_int)
            except Exception:
                return -ic_target
            ic_calc = self.compute_ic(vbe_int, vce_int)
            return ic_calc - ic_target

        vce_lo, vce_hi = 0.001, 5.0
        try:
            err_lo = ic_error(vce_lo)
            err_hi = ic_error(vce_hi)
        except Exception:
            return 0.75, 0.1

        if err_lo * err_hi > 0:
            def equations(x):
                vbe_int, vce_int = x
                vce_int = max(vce_int, 0.001)
                return [self.compute_ic(vbe_int, vce_int) - ic_target,
                        self.compute_ib(vbe_int, vce_int) - ib_target]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sol = fsolve(equations, [0.8, 0.2], full_output=False)
            vbe_int = float(sol[0])
            vce_int = max(float(sol[1]), 0.001)
            vbe_term = vbe_int + ie * self.RE
            vce_term = vce_int + ic_target * self.RC + ie * self.RE
            return vbe_term, max(vce_term, 0.001)

        try:
            vce_int_sol = brentq(ic_error, vce_lo, vce_hi, xtol=1e-8)
        except ValueError:
            return 0.75, 0.1
        vbe_int_sol = self.solve_vbe_for_ib(ib_target, vce_int_sol)
        vbe_term = vbe_int_sol + ie * self.RE
        vce_term = vce_int_sol + ic_target * self.RC + ie * self.RE
        return vbe_term, max(vce_term, 0.001)

    # ==================== 曲线数据生成 ====================

    def ic_vce_curve(self, ib, vce_array):
        """曲线1: 给定 IB，扫描 VCE，返回 IC 数组 (A)。"""
        ic_out = np.zeros_like(vce_array, dtype=float)
        for i, vce in enumerate(vce_array):
            vce_val = max(float(vce), 0.001)
            vbe = self.solve_vbe_for_ib(ib, vce_val)
            ic_out[i] = self.compute_ic(vbe, vce_val)
        return ic_out

    def hfe_vs_ic(self, ic_array, vce=1.0):
        """曲线2: 给定 IC 数组和 VCE，返回 hFE 数组。"""
        hfe_out = np.zeros_like(ic_array, dtype=float)
        for i, ic_target in enumerate(ic_array):
            ic_val = float(ic_target)
            if ic_val < 1e-9:
                hfe_out[i] = 0.0
                continue
            vbe = self.solve_vbe_for_ic(ic_val, vce)
            hfe_out[i] = self.compute_hfe(vbe, vce)
        return hfe_out

    def vbe_sat_vs_ic(self, ic_array, ratio=10.0):
        """曲线3: 饱和区 VBE(sat) vs IC，IC/IB=ratio。"""
        vbe_out = np.zeros_like(ic_array, dtype=float)
        for i, ic_target in enumerate(ic_array):
            ic_val = float(ic_target)
            ib_val = ic_val / ratio
            vbe, _ = self.solve_saturation(ic_val, ib_val)
            vbe_out[i] = vbe
        return vbe_out

    def vce_sat_vs_ic(self, ic_array, ratio=10.0):
        """曲线4: 饱和区 VCE(sat) vs IC，IC/IB=ratio。"""
        vce_out = np.zeros_like(ic_array, dtype=float)
        for i, ic_target in enumerate(ic_array):
            ic_val = float(ic_target)
            ib_val = ic_val / ratio
            _, vce = self.solve_saturation(ic_val, ib_val)
            vce_out[i] = vce
        return vce_out

    def ic_vs_vbe(self, vbe_array, vce=5.0):
        """曲线5: 扫描 VBE，返回 IC 数组 (A)。"""
        ic_out = np.zeros_like(vbe_array, dtype=float)
        for i, vbe in enumerate(vbe_array):
            ic_out[i] = self.compute_ic(float(vbe), vce)
        return ic_out

    def vce_vs_ib(self, ib_array, ic_target):
        """曲线6: 给定 IC，扫描 IB，返回 VCE 数组。
        活跃区用 hFE 近似，饱和区用 solve_saturation。"""
        vce_out = np.zeros_like(ib_array, dtype=float)
        for i, ib in enumerate(ib_array):
            ib_val = float(ib)
            if ib_val < 1e-12:
                vce_out[i] = 5.0
                continue
            ratio = ic_target / ib_val
            if ratio > self.BF * 1.5:
                vce_out[i] = 5.0
                continue
            try:
                _, vce = self.solve_saturation(ic_target, ib_val)
                vce_out[i] = min(vce, 5.0)
            except Exception:
                vce_out[i] = 5.0
        return vce_out


if __name__ == "__main__":
    # 快速验证
    for name, params in PRESETS.items():
        bjt = GummelPoonBJT(params)
        print(f"\n=== {name} ===")
        for ic_ma in [0.1, 1.0, 10.0, 50.0, 100.0]:
            ic = ic_ma * 1e-3
            vbe = bjt.solve_vbe_for_ic(ic, 1.0)
            hfe = bjt.compute_hfe(vbe, 1.0)
            print(f"  IC={ic_ma:6.1f}mA  VBE={vbe:.4f}V  hFE={hfe:.1f}")
        # 饱和区验证
        vbe_s, vce_s = bjt.solve_saturation(10e-3, 1e-3)
        print(f"  VCE(sat)={vce_s:.4f}V @ IC=10mA, IB=1mA")
        print(f"  VBE(sat)={vbe_s:.4f}V @ IC=10mA, IB=1mA")
