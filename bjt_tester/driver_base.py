"""
driver_base.py - 仪器驱动抽象基类
定义所有驱动必须实现的统一接口，实现主程序与硬件的解耦。
"""
from abc import ABC, abstractmethod
from typing import List, Optional


class BaseInstrumentDriver(ABC):
    """
    仪器驱动抽象基类。
    虚拟驱动和真实驱动均继承此类，保证接口一致。
    """

    @abstractmethod
    def connect(self) -> bool:
        """建立与仪器的连接，成功返回 True"""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """断开与仪器的连接，关闭所有输出"""
        ...

    @abstractmethod
    def set_voltage(self, channel: int, voltage: float) -> None:
        """
        设置指定通道的输出电压（DC 模式）。
        对应 Model3 AWG 的 DC 输出功能。
        参数:
            channel: 通道号 (0 或 1)
            voltage: 输出电压值 (V)
        """
        ...

    @abstractmethod
    def read_voltage(self, channel: int, samples: int = 2048) -> List[float]:
        """
        读取指定通道的电压采样数据。
        对应 Model3 ADC 的采集功能。
        参数:
            channel: 通道号 (0 或 1)
            samples: 采样点数，默认 2048
        返回:
            采样数据列表
        """
        ...

    @abstractmethod
    def set_current_source(self, channel: int, current: float) -> None:
        """
        设置指定通道的电流源输出。
        通过外部电阻将电压转换为电流。
        参数:
            channel: 通道号
            current: 目标电流值 (A)
        """
        ...

    @abstractmethod
    def read_current(self, channel: int, samples: int = 2048) -> List[float]:
        """
        读取指定通道的电流采样数据。
        通过跨阻放大器或采样电阻将电流转换为电压后采集。
        参数:
            channel: 通道号
            samples: 采样点数
        返回:
            电流数据列表 (A)
        """
        ...

    @abstractmethod
    def set_power_supply(self, channel: int, voltage: float,
                         enable: bool = True) -> None:
        """
        控制程控电源输出。
        对应 Model3 的正/负电源通道。
        参数:
            channel: 0=正电源(0~5V), 1=负电源(-5~0V)
            voltage: 输出电压值 (V)
            enable:  是否使能输出
        """
        ...

    @abstractmethod
    def dmm_read_dc_voltage(self) -> float:
        """使用 DMM 测量直流电压 (V)"""
        ...

    @abstractmethod
    def dmm_read_diode(self) -> float:
        """使用 DMM 二极管模式测量正向压降 (V)"""
        ...

    @abstractmethod
    def set_digital_output(self, channel: int, state: bool) -> None:
        """设置数字 IO 输出状态，用于继电器切换测试电路"""
        ...

    @abstractmethod
    def read_digital_input(self) -> int:
        """读取数字 IO 输入状态，返回 16 位状态字"""
        ...

    @abstractmethod
    def emergency_stop(self) -> None:
        """紧急停止：立即关闭所有输出通道"""
        ...
