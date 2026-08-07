"""文档四角检测与透视裁剪模块。

算法：阈值分割（Otsu / 自适应阈值）得到文档掩码，取最大连通区域，
再经多边形逼近 / 凸包 / 最小外接矩形提取四边形；文档铺满画面时直接整幅。
"""

import cv2
import numpy as np


def order_points(pts):
    """按 左上、右上、右下、左下 排序四个角点。"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def is_valid_quad(pts, img_h, img_w):
    """校验四边形：四条边不能过短，且必须是凸四边形。"""
    sides = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
    if min(sides) < 0.02 * max(img_h, img_w):
        return False
    if not cv2.isContourConvex(pts.astype(np.int32).reshape(-1, 1, 2)):
        return False
    return True


def _quad_from_contour(contour, img_h, img_w, min_area):
    """从轮廓提取四边形角点：多边形逼近 -> 凸包。返回 (面积, 角点) 或 None。"""
    peri = cv2.arcLength(contour, True)
    for eps in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            pts = order_points(approx.reshape(4, 2).astype("float32"))
            if area > min_area and is_valid_quad(pts, img_h, img_w):
                return area, pts
            break
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area > min_area:
        hull_peri = cv2.arcLength(hull, True)
        hull_approx = cv2.approxPolyDP(hull, 0.02 * hull_peri, True)
        if len(hull_approx) == 4:
            pts = order_points(hull_approx.reshape(4, 2).astype("float32"))
            if is_valid_quad(pts, img_h, img_w):
                return hull_area, pts
    return None


def detect_document_corners(frame, min_area_ratio=0.03, max_process_width=960):
    """
    在画面中定位文档的四个角点。

    参数:
        frame: BGR 图像
        min_area_ratio: 四边形最小面积占画面比例
        max_process_width: 处理时缩放到的最大宽度（提升速度）

    返回:
        排序后的 (4, 2) float32 角点数组；未检测到时返回 None
    """
    h, w = frame.shape[:2]
    scale = 1.0
    work = frame
    if w > max_process_width:
        scale = max_process_width / w
        work = cv2.resize(frame, (max_process_width, int(h * scale)),
                          interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    min_area = img_w * img_h * min_area_ratio
    kernel7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    best = None
    full_frame_seen = False
    all_contours = []

    def consider(contour, allow_full=False):
        nonlocal best, full_frame_seen
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        is_full = cw >= 0.97 * img_w and ch >= 0.97 * img_h
        if is_full:
            full_frame_seen = True
            if not allow_full:
                return
            pts = np.float32([[0, 0], [img_w - 1, 0],
                              [img_w - 1, img_h - 1], [0, img_h - 1]])
            if best is None or area > best[0]:
                best = (area, pts)
            return
        res = _quad_from_contour(contour, img_h, img_w, min_area)
        if res and (best is None or res[0] > best[0]):
            best = (res[0], res[1])

    # 阈值分割管线（含内部孔洞，兼容“白底上放文档”的拍摄）
    otsu = cv2.threshold(blur, 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(blur, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 15)
    for mask in (otsu, adaptive):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel7)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP,
                                               cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        hierarchy = hierarchy[0]
        outer = [c for i, c in enumerate(contours) if hierarchy[i][3] == -1]
        inner = [c for i, c in enumerate(contours) if hierarchy[i][3] != -1]
        all_contours.extend(outer)
        for c in sorted(outer, key=cv2.contourArea, reverse=True)[:4]:
            consider(c)
        for c in sorted(inner, key=cv2.contourArea, reverse=True)[:4]:
            consider(c)

    # Canny 边缘管线（处理纹理复杂的背景）
    edged = cv2.Canny(blur, 75, 200)
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel5)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        all_contours.extend(contours)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:4]:
            consider(c)

    # 文档铺满画面时直接使用整幅
    if best is None and full_frame_seen:
        best = (img_w * img_h,
                np.float32([[0, 0], [img_w - 1, 0],
                            [img_w - 1, img_h - 1], [0, img_h - 1]]))

    # 终极兜底：最大轮廓的最小外接矩形（适合固定俯拍）
    if best is None and all_contours:
        largest = max(all_contours, key=cv2.contourArea)
        box = cv2.boxPoints(cv2.minAreaRect(largest))
        area = cv2.contourArea(box)
        bw = np.linalg.norm(box[0] - box[1])
        bh = np.linalg.norm(box[1] - box[2])
        if (area > min_area and min(bw, bh) > 0.06 * max(img_w, img_h)
                and 0.2 <= max(bw, bh) / max(min(bw, bh), 1.0) <= 5):
            pts = order_points(box.astype("float32"))
            if is_valid_quad(pts, img_h, img_w):
                best = (area, pts)

    if best is None:
        return None
    _, pts = best
    if scale != 1.0:
        pts = pts / scale
    return pts


def four_point_transform(image, pts):
    """根据四个角点做透视变换，返回校正后的文档图像。"""
    rect = order_points(pts)
    tl, tr, br, bl = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))
    dst = np.array([[0, 0],
                    [max_width - 1, 0],
                    [max_width - 1, max_height - 1],
                    [0, max_height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def crop_document(frame):
    """检测文档并裁剪；检测失败返回 (None, None)。"""
    pts = detect_document_corners(frame)
    if pts is None:
        return None, None
    return four_point_transform(frame, pts), pts
