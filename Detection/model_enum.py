from enum import Enum

class ModelType(str, Enum):
    """模型类型枚举"""
    UNET2D = "unet2d"
    ATTENTION_UNET2D = "attention_unet2d"