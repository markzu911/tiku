"""
图片增强模块 - 在 OCR 和视觉分析前提升图片质量
"""
from io import BytesIO

from PIL import Image, ImageEnhance, ImageFilter


def enhance_image_for_analysis(image_bytes: bytes) -> bytes:
    """
    增强图片以提高 OCR 和视觉分析准确率

    处理步骤：
    1. 自动调整对比度
    2. 提升清晰度
    3. 轻微锐化
    4. 降噪处理

    Args:
        image_bytes: 原始图片字节

    Returns:
        增强后的图片字节 (PNG 格式)
    """
    try:
        image = Image.open(BytesIO(image_bytes))

        # 转换为 RGB 模式
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 1. 自动对比度增强 (1.3 倍，适度增强黑白对比)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.3)

        # 2. 亮度微调 (1.1 倍，略微提亮)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)

        # 3. 清晰度增强 (1.5 倍)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)

        # 4. 锐化滤镜 (增强边缘)
        image = image.filter(ImageFilter.SHARPEN)

        # 5. 轻微降噪 (平滑处理，减少噪点)
        image = image.filter(ImageFilter.SMOOTH_MORE)

        # 输出为高质量 PNG
        output = BytesIO()
        image.save(output, format="PNG", optimize=True)

        return output.getvalue()

    except Exception as exc:
        # 如果增强失败，返回原始图片
        return image_bytes


def enhance_image_aggressive(image_bytes: bytes) -> bytes:
    """
    激进的图片增强 - 用于严重模糊或低质量图片

    处理步骤：
    1. 高强度对比度调整
    2. 大幅度清晰度提升
    3. 多次锐化
    4. 边缘增强

    Args:
        image_bytes: 原始图片字节

    Returns:
        增强后的图片字节 (PNG 格式)
    """
    try:
        image = Image.open(BytesIO(image_bytes))

        if image.mode != "RGB":
            image = image.convert("RGB")

        # 1. 强对比度 (1.5 倍)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)

        # 2. 亮度调整 (1.2 倍)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.2)

        # 3. 高清晰度 (2.0 倍)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)

        # 4. 双重锐化
        image = image.filter(ImageFilter.SHARPEN)
        image = image.filter(ImageFilter.SHARPEN)

        # 5. 边缘增强
        image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)

        output = BytesIO()
        image.save(output, format="PNG", optimize=True)

        return output.getvalue()

    except Exception as exc:
        return image_bytes
