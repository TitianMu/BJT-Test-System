"""
driver_factory.py - 驱动工厂类
根据配置创建虚拟或真实驱动实例，实现一行代码切换模式。
"""
from driver_base import BaseInstrumentDriver


class DriverFactory:
    """
    驱动工厂：根据模式字符串创建对应的驱动实例。
    支持模式: "virtual" (默认), "real"
    """

    @staticmethod
    def create(mode: str = "virtual") -> BaseInstrumentDriver:
        """
        创建驱动实例。
        参数:
            mode: "virtual" 使用虚拟驱动, "real" 使用真实驱动
        返回:
            BaseInstrumentDriver 子类实例
        """
        if mode == "virtual":
            from virtual_driver import VirtualInstrumentDriver
            return VirtualInstrumentDriver()
        elif mode == "real":
            from real_driver import RealInstrumentDriver
            return RealInstrumentDriver()
        else:
            raise ValueError(
                f"未知驱动模式: {mode}，支持 'virtual' 或 'real'"
            )

    @staticmethod
    def available_modes():
        """返回所有可用的驱动模式"""
        return ["virtual", "real"]
