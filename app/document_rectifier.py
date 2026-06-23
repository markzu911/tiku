"""
试卷矫正模块 - 将手机拍摄的试卷图矫正成标准扫描图

功能：
1. 检测试卷边界（四个角点）
2. 透视变换矫正成标准矩形
3. 旋转矫正（确保试卷方向正确）
4. 裁剪去除背景、桌面、其他页面
"""
import math
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    排序四个角点：左上、右上、右下、左下

    Args:
        pts: 4x2 数组，包含四个角点坐标

    Returns:
        排序后的角点数组
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    # 左上角点的 x+y 最小，右下角点的 x+y 最大
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 右上角点的 y-x 最小，左下角点的 y-x 最大
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    透视变换：将四边形矫正为矩形

    Args:
        image: 原始图片 (numpy array)
        pts: 四个角点坐标

    Returns:
        矫正后的图片
    """
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算新图片的宽度
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # 计算新图片的高度
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # 目标矩形的四个角点
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    # 计算透视变换矩阵
    # 使用简单的仿射变换近似
    img_pil = Image.fromarray(image)

    # 使用 PIL 的 transform 进行透视变换
    coeffs = _find_coeffs(dst, rect)
    warped = img_pil.transform(
        (max_width, max_height),
        Image.PERSPECTIVE,
        coeffs,
        Image.BICUBIC
    )

    return np.array(warped)


def _find_coeffs(target_pts: np.ndarray, source_pts: np.ndarray) -> tuple:
    """
    计算透视变换系数

    Args:
        target_pts: 目标点 (4x2)
        source_pts: 源点 (4x2)

    Returns:
        8个变换系数
    """
    matrix = []
    for s, t in zip(source_pts, target_pts):
        matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
        matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])

    A = np.matrix(matrix, dtype=np.float32)
    B = np.array(source_pts).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def _detect_document_contour(image: Image.Image) -> Optional[np.ndarray]:
    """
    检测试卷轮廓（四个角点）

    Args:
        image: PIL Image 对象

    Returns:
        四个角点坐标，如果检测失败返回 None
    """
    # 转换为灰度图
    gray = image.convert('L')

    # 缩小图片加速处理
    orig_width, orig_height = gray.size
    scale = 1000 / max(orig_width, orig_height)
    if scale < 1:
        new_size = (int(orig_width * scale), int(orig_height * scale))
        gray = gray.resize(new_size, Image.LANCZOS)
    else:
        scale = 1.0

    # 转为 numpy 数组
    img_array = np.array(gray)

    # 高斯模糊
    blurred = Image.fromarray(img_array).filter(ImageFilter.GaussianBlur(radius=5))
    blurred_array = np.array(blurred)

    # Canny 边缘检测 (手动实现简化版)
    # 使用 Sobel 算子
    edges = _simple_edge_detection(blurred_array)

    # 查找轮廓 (简化版：查找最大连通区域的外接矩形)
    contours = _find_largest_contour(edges)

    if contours is None:
        return None

    # 将坐标还原到原始尺寸
    contours = contours / scale

    return contours


def _simple_edge_detection(image: np.ndarray, threshold: int = 50) -> np.ndarray:
    """
    简单的边缘检测（基于梯度）

    Args:
        image: 灰度图数组
        threshold: 边缘阈值

    Returns:
        边缘二值图
    """
    # Sobel 算子
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    h, w = image.shape
    grad_x = np.zeros_like(image, dtype=np.float32)
    grad_y = np.zeros_like(image, dtype=np.float32)

    # 卷积计算梯度
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            patch = image[i-1:i+2, j-1:j+2]
            grad_x[i, j] = np.sum(patch * sobel_x)
            grad_y[i, j] = np.sum(patch * sobel_y)

    # 计算梯度幅值
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # 二值化
    edges = (magnitude > threshold).astype(np.uint8) * 255

    return edges


def _find_largest_contour(edges: np.ndarray) -> Optional[np.ndarray]:
    """
    查找最大轮廓的四个角点

    Args:
        edges: 边缘二值图

    Returns:
        四个角点坐标，如果失败返回 None
    """
    h, w = edges.shape

    # 查找边缘点
    edge_points = np.column_stack(np.where(edges > 0))

    if len(edge_points) < 100:
        return None

    # 使用边缘点找到外接矩形的四个角
    # 简化方法：找到 x+y 最小、最大和 x-y 最小、最大的点
    y_coords, x_coords = edge_points[:, 0], edge_points[:, 1]

    # 四个角点候选
    sum_coords = x_coords + y_coords
    diff_coords = x_coords - y_coords

    tl_idx = np.argmin(sum_coords)
    br_idx = np.argmax(sum_coords)
    tr_idx = np.argmax(diff_coords)
    bl_idx = np.argmin(diff_coords)

    corners = np.array([
        [x_coords[tl_idx], y_coords[tl_idx]],
        [x_coords[tr_idx], y_coords[tr_idx]],
        [x_coords[br_idx], y_coords[br_idx]],
        [x_coords[bl_idx], y_coords[bl_idx]]
    ], dtype=np.float32)

    # 验证是否形成合理的四边形
    # 计算面积，如果太小说明检测失败
    area = _polygon_area(corners)
    image_area = h * w

    if area < image_area * 0.1:  # 面积小于图片的 10%
        return None

    return corners


def _polygon_area(points: np.ndarray) -> float:
    """计算多边形面积（鞋带公式）"""
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _is_trustworthy_document_contour(points: np.ndarray, image_size: tuple[int, int]) -> bool:
    """Reject weak contour guesses before they can distort a valid worksheet photo."""
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return False

    width, height = image_size
    if width <= 0 or height <= 0:
        return False

    tl, tr, br, bl = _order_points(points)
    normalized_corners = np.array(
        [
            [tl[0] / width, tl[1] / height],
            [tr[0] / width, tr[1] / height],
            [br[0] / width, br[1] / height],
            [bl[0] / width, bl[1] / height],
        ]
    )

    # A document contour must place each corner in its corresponding image quadrant.
    # This prevents isolated background edges from becoming a fake paper boundary.
    if not (
        normalized_corners[0, 0] <= 0.4
        and normalized_corners[0, 1] <= 0.4
        and normalized_corners[1, 0] >= 0.6
        and normalized_corners[1, 1] <= 0.4
        and normalized_corners[2, 0] >= 0.6
        and normalized_corners[2, 1] >= 0.6
        and normalized_corners[3, 0] <= 0.4
        and normalized_corners[3, 1] >= 0.6
    ):
        return False

    top = np.linalg.norm(tr - tl)
    right = np.linalg.norm(br - tr)
    bottom = np.linalg.norm(bl - br)
    left = np.linalg.norm(tl - bl)
    edge_lengths = (top, right, bottom, left)
    diagonal = math.hypot(width, height)
    if min(edge_lengths) < diagonal * 0.05:
        return False

    # Perspective may shorten an opposite edge, but not collapse it into a line.
    if min(top, bottom) / max(top, bottom) < 0.35:
        return False
    if min(left, right) / max(left, right) < 0.35:
        return False

    return _polygon_area(np.array([tl, tr, br, bl])) >= width * height * 0.2


def _detect_rotation(image: Image.Image) -> float:
    """
    检测图片旋转角度（基于文本行倾斜检测）

    Args:
        image: PIL Image 对象

    Returns:
        旋转角度（度数），需要顺时针旋转的角度
    """
    try:
        # 转换为灰度图
        gray = image.convert('L')

        # 缩小图片加速处理
        orig_width, orig_height = gray.size
        scale = 800 / max(orig_width, orig_height)
        if scale < 1:
            new_size = (int(orig_width * scale), int(orig_height * scale))
            gray = gray.resize(new_size, Image.LANCZOS)

        # 转为 numpy 数组
        img_array = np.array(gray)

        # 边缘检测
        edges = _simple_edge_detection(img_array, threshold=100)

        # Hough 直线检测（简化版）
        angles = _detect_lines(edges)

        if len(angles) == 0:
            return 0.0

        # 计算中位数角度（更鲁棒）
        median_angle = np.median(angles)

        # 限制在 -45 到 45 度之间
        if median_angle > 45:
            median_angle -= 90
        elif median_angle < -45:
            median_angle += 90

        # 小于 0.5 度不矫正
        if abs(median_angle) < 0.5:
            return 0.0

        return float(median_angle)

    except Exception:
        return 0.0


def _detect_lines(edges: np.ndarray) -> list[float]:
    """
    简化的 Hough 直线检测，返回检测到的直线角度列表

    Args:
        edges: 边缘二值图

    Returns:
        角度列表（度数）
    """
    h, w = edges.shape
    edge_points = np.column_stack(np.where(edges > 0))

    if len(edge_points) < 50:
        return []

    # 随机采样点对来估计直线
    angles = []
    num_samples = min(200, len(edge_points) // 2)

    np.random.seed(42)  # 固定随机种子
    for _ in range(num_samples):
        # 随机选择两个点
        idx = np.random.choice(len(edge_points), 2, replace=False)
        p1, p2 = edge_points[idx]

        # 计算角度
        dy = p2[0] - p1[0]
        dx = p2[1] - p1[1]

        # 忽略距离太近的点对
        if abs(dx) < 10 and abs(dy) < 10:
            continue

        angle = math.degrees(math.atan2(dy, dx))

        # 只关注接近水平的线（文本行）
        if -45 < angle < 45:
            angles.append(angle)

    return angles


def rectify_document(image_bytes: bytes) -> tuple[bytes, bool]:
    """
    矫正试卷图片

    Args:
        image_bytes: 原始图片字节

    Returns:
        (矫正后的图片字节, 是否需要手动裁剪)
        如果检测失败，返回 (原图, True)
    """
    try:
        image = Image.open(BytesIO(image_bytes))

        # 转换为 RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 检测试卷边界
        corners = _detect_document_contour(image)

        if corners is None or not _is_trustworthy_document_contour(corners, image.size):
            # 检测失败，返回原图并标记需要手动裁剪
            return image_bytes, True

        # 透视变换矫正
        img_array = np.array(image)
        warped = _four_point_transform(img_array, corners)

        # 旋转矫正
        rectified = Image.fromarray(warped)
        rotation_angle = _detect_rotation(rectified)

        if abs(rotation_angle) > 0.5:
            # 顺时针旋转，expand=True 确保不裁剪
            rectified = rectified.rotate(-rotation_angle, expand=True, fillcolor=(255, 255, 255))

        # 自动裁剪边缘空白
        rectified = ImageOps.crop(rectified, border=10)

        # 输出为 PNG
        output = BytesIO()
        rectified.save(output, format='PNG', optimize=True)

        return output.getvalue(), False

    except Exception:
        # 任何异常都返回原图并标记需要手动裁剪
        return image_bytes, True
