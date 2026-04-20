from .device_base import DeviceBase
import os

_DISABLE_KEYBOARD = os.environ.get("LEHOME_DISABLE_KEYBOARD") == "1"

if not _DISABLE_KEYBOARD:
    from .lerobot import SO101Leader, BiSO101Leader
    from .keyboard import Se3Keyboard, BiKeyboard

__all__ = ["DeviceBase"]

if not _DISABLE_KEYBOARD:
    __all__.extend([
        "SO101Leader",
        "BiSO101Leader",
        "Se3Keyboard",
        "BiKeyboard",
    ])
    # "XboxController",  # Commented out as it may not exist
