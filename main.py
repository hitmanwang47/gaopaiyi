"""高拍仪应用：选择摄像头 -> 实时预览(自动框出文档) -> 拍摄 -> 记录列表与缩略图。"""

import os
import subprocess
import sys
import time

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QListWidget, QListWidgetItem, QHBoxLayout, QVBoxLayout, QGridLayout,
    QSplitter, QCheckBox, QFileDialog, QMessageBox, QGroupBox, QSlider,
    QAbstractItemView)

from detector import crop_document, detect_document_corners, four_point_transform

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(APP_DIR, "captures")
COMMON_RESOLUTIONS = [
    (160, 120), (320, 240), (480, 360), (640, 360), (640, 480),
    (720, 480), (720, 576), (800, 600), (960, 540), (1024, 768),
    (1152, 864), (1280, 720), (1280, 800), (1280, 960), (1280, 1024),
    (1366, 768), (1440, 900), (1600, 900), (1600, 1200), (1680, 1050),
    (1920, 1080), (1920, 1200), (2048, 1536), (2560, 1440), (3264, 2448),
    (3840, 2160),
]


def _camera_backend():
    """Windows 用 DirectShow，Linux/macOS 用系统默认后端。"""
    return cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY


def _open_with_default_app(path):
    """用系统默认程序打开文件或目录（跨平台）。"""
    if os.name == "nt":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


# ---------- 摄像头枚举 ----------
def list_cameras(max_index=9):
    """探测可用摄像头，返回 [(索引, 显示名)]。"""
    available = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, _camera_backend())
        if not cap.isOpened():
            cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append((i, "摄像头 %d" % (i + 1)))
            cap.release()
    return available


def list_resolutions(camera_index):
    """探测摄像头支持的常见分辨率列表（实际读帧确认生效）。"""
    supported = []
    cap = cv2.VideoCapture(camera_index, _camera_backend())
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return supported
    try:
        for w, h in COMMON_RESOLUTIONS:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            fh, fw = frame.shape[:2]
            if (fw, fh) == (w, h) and (w, h) not in supported:
                supported.append((w, h))
    finally:
        cap.release()
    return supported


def _probe_range(cap, prop):
    """探测摄像头某个属性（亮度/对比度/饱和度）的实际取值范围。"""
    values = []
    for v in (0, 64, 128, 192, 255):
        cap.set(prop, v)
        time.sleep(0.03)
        gv = cap.get(prop)
        if gv is not None and gv >= 0:
            values.append(gv)
    if len(values) < 2:
        return None
    lo, hi = min(values), max(values)
    if hi - lo < 2:
        return None
    return lo, hi


# ---------- 摄像头工作线程 ----------
class CameraWorker(QThread):
    """后台线程：读取摄像头画面并做文档角点检测。"""

    frame_ready = pyqtSignal(object)      # (frame_bgr, corners, stable_corners)
    camera_error = pyqtSignal(str)
    properties_ready = pyqtSignal(object)  # 实际生效的摄像头参数

    def __init__(self, camera_index, resolution=None, brightness=None,
                 contrast=None, saturation=None, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.resolution = resolution
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self._running = True
        self.last_good = None
        self.stable = None
        self.miss = 0

    @staticmethod
    def _corners_close(a, b, frame_shape):
        """判断两帧角点是否基本一致（中心位移小、面积相近）。"""
        h, w = int(frame_shape.shape[0]), int(frame_shape.shape[1])
        center_a = a.mean(axis=0)
        center_b = b.mean(axis=0)
        dist = np.linalg.norm(center_a - center_b) / max(h, w)
        area_a = cv2.contourArea(a)
        area_b = cv2.contourArea(b)
        ratio = max(area_a, area_b) / max(min(area_a, area_b), 1.0)
        return dist < 0.06 and ratio < 1.6

    def run(self):
        cap = cv2.VideoCapture(self.camera_index, _camera_backend())
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.camera_error.emit("无法打开摄像头（索引 %d）" % self.camera_index)
            return
        if self.resolution:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        # 关闭自动曝光/白平衡/对焦，避免手动参数被自动模式覆盖
        for auto_prop in (cv2.CAP_PROP_AUTO_EXPOSURE, cv2.CAP_PROP_AUTO_WB,
                          cv2.CAP_PROP_AUTO_FOCUS, cv2.CAP_PROP_AUTO_SHARPNESS):
            cap.set(auto_prop, 0)

        original = {
            "brightness": cap.get(cv2.CAP_PROP_BRIGHTNESS),
            "contrast": cap.get(cv2.CAP_PROP_CONTRAST),
            "saturation": cap.get(cv2.CAP_PROP_SATURATION),
        }
        ranges = {
            "brightness": _probe_range(cap, cv2.CAP_PROP_BRIGHTNESS),
            "contrast": _probe_range(cap, cv2.CAP_PROP_CONTRAST),
            "saturation": _probe_range(cap, cv2.CAP_PROP_SATURATION),
        }
        # 恢复探测前的参数，避免探测过程改变画面
        for name, prop in (("brightness", cv2.CAP_PROP_BRIGHTNESS),
                           ("contrast", cv2.CAP_PROP_CONTRAST),
                           ("saturation", cv2.CAP_PROP_SATURATION)):
            ov = original.get(name)
            if ov is not None and ov >= 0:
                cap.set(prop, ov)

        actual = dict(original)
        for name, prop in (("brightness", cv2.CAP_PROP_BRIGHTNESS),
                           ("contrast", cv2.CAP_PROP_CONTRAST),
                           ("saturation", cv2.CAP_PROP_SATURATION)):
            value = getattr(self, name)
            rng = ranges.get(name)
            if value is None or rng is None:
                continue
            value = max(rng[0], min(rng[1], float(value)))
            cap.set(prop, value)
            actual[name] = cap.get(prop)
        actual["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.properties_ready.emit({"ranges": ranges, "actual": actual})

        while self._running:
            ok, frame = cap.read()
            if not ok:
                continue
            corners = detect_document_corners(frame)
            if corners is not None:
                if (self.last_good is not None
                        and self._corners_close(corners, self.last_good,
                                                frame)):
                    self.stable = corners
                    self.miss = 0
                else:
                    if self.stable is None:
                        self.stable = corners
                    self.miss = 0
                self.last_good = corners
            else:
                self.miss += 1
                if self.miss > 12:      # 连续约 0.4 秒丢失则清除框
                    self.stable = None
            self.frame_ready.emit((frame, corners, self.stable))
        cap.release()

    def stop(self):
        self._running = False
        self.wait()


class ResolutionProbe(QThread):
    """后台探测摄像头支持的分辨率。"""

    probe_done = pyqtSignal(object)

    def __init__(self, camera_index, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index

    def run(self):
        self.probe_done.emit(list_resolutions(self.camera_index))


class PreviewLabel(QLabel):
    """可感知鼠标拖动的预览控件。"""

    corner_pressed = pyqtSignal(float, float)
    corner_moved = pyqtSignal(float, float)
    corner_released = pyqtSignal()

    def mousePressEvent(self, event):
        self.corner_pressed.emit(event.x(), event.y())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.corner_moved.emit(event.x(), event.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.corner_released.emit()
        super().mouseReleaseEvent(event)



class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("高拍仪 - 文档拍摄")
        self.resize(1180, 760)

        self.worker = None
        self.probe = None
        self.latest_frame = None
        self.latest_corners = None
        self.save_counter = 0
        self.paused = False
        self.frozen_frame = None
        self.manual_corners = None
        self._drag_corner = None
        self._display_w = 0
        self._display_h = 0
        self.output_dir = DEFAULT_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

        self._sliders_touched = False
        self._settings_timer = QTimer(self)
        self._settings_timer.setSingleShot(True)
        self._settings_timer.setInterval(400)
        self._settings_timer.timeout.connect(self.restart_preview)

        self._build_ui()
        self.refresh_cameras()
        self.statusBar().showMessage("请选择摄像头并点击“开始预览”")

    # ---------- 暂停 / 手动调整 ----------
    def toggle_pause(self):
        if self.worker is None:
            return
        if not self.paused:
            self.paused = True
            self.pause_btn.setText("继续")
            self.frozen_frame = (self.latest_frame.copy()
                                 if self.latest_frame is not None else None)
            self.manual_corners = (self.latest_corners.copy()
                                   if self.latest_corners is not None else None)
            self._drag_corner = None
            self._draw_frozen()
            self.statusBar().showMessage(
                "已暂停：拖动四个红点微调边框，再点“拍摄”或“继续”")
        else:
            self.paused = False
            self.pause_btn.setText("暂停")
            self.manual_corners = None
            self.frozen_frame = None
            self._drag_corner = None
            self.statusBar().showMessage("已恢复自动检测")

    def _draw_frozen(self):
        if self.frozen_frame is None:
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("预览画面")
            return
        display = self.frozen_frame.copy()
        if self.manual_corners is not None:
            pts = np.int32(self.manual_corners.reshape(-1, 1, 2))
            cv2.polylines(display, [pts], True, (0, 255, 0), 3)
            for pt in pts:
                cv2.circle(display, tuple(pt[0]), 10, (0, 0, 255), -1)
        self._show_frame(display)

    def _widget_to_image(self, x, y):
        pixmap = self.preview_label.pixmap()
        if pixmap is None or pixmap.isNull() or self._display_w == 0:
            return None
        label = self.preview_label
        pm_size = pixmap.size()
        x0 = (label.width() - pm_size.width()) / 2.0
        y0 = (label.height() - pm_size.height()) / 2.0
        return ((x - x0) * self._display_w / pm_size.width(),
                (y - y0) * self._display_h / pm_size.height())

    def _nearest_corner(self, ix, iy):
        if self.manual_corners is None:
            return None
        threshold = max(15, 0.03 * min(self._display_h, self._display_w))
        best_idx = None
        best_dist = threshold
        for i, (cx, cy) in enumerate(self.manual_corners):
            d = ((cx - ix) ** 2 + (cy - iy) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def on_preview_press(self, x, y):
        if not self.paused or self.manual_corners is None:
            return
        pos = self._widget_to_image(x, y)
        if pos is None:
            return
        corner = self._nearest_corner(*pos)
        if corner is not None:
            self._drag_corner = corner
            self._set_corner(corner, *pos)

    def on_preview_move(self, x, y):
        if self._drag_corner is None:
            return
        pos = self._widget_to_image(x, y)
        if pos is None:
            return
        self._set_corner(self._drag_corner, *pos)

    def on_preview_release(self):
        self._drag_corner = None

    def _set_corner(self, index, ix, iy):
        if self.manual_corners is None:
            return
        corners = self.manual_corners.copy()
        corners[index] = [float(np.clip(ix, 0, self._display_w - 1)),
                          float(np.clip(iy, 0, self._display_h - 1))]
        self.manual_corners = corners
        self._draw_frozen()

    # ---------- UI ----------
    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # 顶部控制栏
        controls = QGridLayout()
        controls.addWidget(QLabel("摄像头:"), 0, 0)
        self.camera_combo = QComboBox()
        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)
        controls.addWidget(self.camera_combo, 0, 1)
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.refresh_cameras)
        controls.addWidget(self.refresh_btn, 0, 2)
        self.preview_btn = QPushButton("开始预览")
        self.preview_btn.clicked.connect(self.toggle_preview)
        controls.addWidget(self.preview_btn, 0, 3)
        self.capture_btn = QPushButton("拍摄")
        self.capture_btn.setEnabled(False)
        self.capture_btn.clicked.connect(self.capture)
        controls.addWidget(self.capture_btn, 0, 4)
        controls.addWidget(QLabel("保存格式:"), 0, 5)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["JPG", "PNG"])
        controls.addWidget(self.format_combo, 0, 6)
        self.auto_crop_check = QCheckBox("自动裁剪文档")
        self.auto_crop_check.setChecked(True)
        controls.addWidget(self.auto_crop_check, 0, 7)
        self.dir_btn = QPushButton("保存目录")
        self.dir_btn.clicked.connect(self.choose_dir)
        controls.addWidget(self.dir_btn, 0, 8)
        self.open_dir_btn = QPushButton("打开目录")
        self.open_dir_btn.clicked.connect(self.open_dir)
        controls.addWidget(self.open_dir_btn, 0, 9)
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.toggle_pause)
        controls.addWidget(self.pause_btn, 0, 10)
        controls.setColumnStretch(11, 1)
        root.addLayout(controls)

        # 摄像头设置区
        settings = QGroupBox("摄像头设置")
        grid = QGridLayout(settings)
        grid.addWidget(QLabel("分辨率:"), 0, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItem("自动", None)
        self.resolution_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.resolution_combo.setMinimumWidth(130)
        self.resolution_combo.currentIndexChanged.connect(self.on_setting_changed)
        grid.addWidget(self.resolution_combo, 0, 1)
        grid.addWidget(QLabel("亮度:"), 0, 2)
        self.brightness_slider = self._make_slider(grid, 0, 3, "brightness")
        grid.addWidget(QLabel("对比度:"), 0, 4)
        self.contrast_slider = self._make_slider(grid, 0, 5, "contrast")
        grid.addWidget(QLabel("饱和度:"), 0, 6)
        self.saturation_slider = self._make_slider(grid, 0, 7, "saturation")
        grid.setColumnStretch(8, 1)
        root.addWidget(settings)

        # 中间：预览 + 记录列表
        splitter = QSplitter(Qt.Horizontal)
        self.preview_label = PreviewLabel("预览画面")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setStyleSheet(
            "background:#202020; color:#888; font-size:18px;")
        self.preview_label.corner_pressed.connect(self.on_preview_press)
        self.preview_label.corner_moved.connect(self.on_preview_move)
        self.preview_label.corner_released.connect(self.on_preview_release)
        splitter.addWidget(self.preview_label)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("拍摄记录（双击查看原图）"))
        self.record_list = QListWidget()
        self.record_list.setViewMode(QListWidget.IconMode)
        self.record_list.setResizeMode(QListWidget.Adjust)
        self.record_list.setIconSize(QPixmap(180, 135).size())
        self.record_list.setGridSize(QPixmap(200, 175).size())
        self.record_list.setWordWrap(True)
        self.record_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.record_list.itemDoubleClicked.connect(self.open_record)
        self.clear_btn = QPushButton("清空列表")
        self.clear_btn.clicked.connect(self.record_list.clear)
        right_layout.addWidget(self.record_list)
        right_layout.addWidget(self.clear_btn)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

    def _make_slider(self, grid, row, col, name):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 255)
        slider.setValue(128)
        slider.valueChanged.connect(self.on_setting_changed)
        slider.setToolTip(name)
        grid.addWidget(slider, row, col)
        return slider

    # ---------- 摄像头 ----------
    def refresh_cameras(self):
        if self.worker is not None:
            self.stop_preview()
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        cameras = list_cameras()
        for idx, name in cameras:
            self.camera_combo.addItem(name, idx)
        self.camera_combo.blockSignals(False)
        if not cameras:
            self.camera_combo.addItem("未检测到摄像头", -1)
            self.resolution_combo.clear()
            self.resolution_combo.addItem("自动", None)
            self.statusBar().showMessage("未检测到摄像头")
        else:
            self.on_camera_changed()

    def on_camera_changed(self):
        idx = self.camera_combo.currentData()
        if idx is None or idx < 0:
            return
        self._sliders_touched = False
        self.resolution_combo.blockSignals(True)
        self.resolution_combo.clear()
        self.resolution_combo.addItem("自动", None)
        self.resolution_combo.blockSignals(False)
        # 后台探测分辨率
        if self.probe is not None:
            self.probe.wait()
        self.probe = ResolutionProbe(idx)
        self.probe.probe_done.connect(self.on_resolutions_ready)
        self.probe.start()
        if self.worker is not None:
            self.restart_preview()

    def on_resolutions_ready(self, resolutions):
        self.resolution_combo.blockSignals(True)
        self.resolution_combo.clear()
        self.resolution_combo.addItem("自动", None)
        for w, h in resolutions:
            self.resolution_combo.addItem("%d x %d" % (w, h), (w, h))
        self.resolution_combo.blockSignals(False)

    def on_setting_changed(self):
        self._sliders_touched = True
        self._settings_timer.start()

    # ---------- 预览 ----------
    def toggle_preview(self):
        if self.worker is not None:
            self.stop_preview()
            return
        idx = self.camera_combo.currentData()
        if idx is None or idx < 0:
            QMessageBox.warning(self, "提示", "未检测到可用的摄像头")
            return
        self.start_preview(idx)

    def _current_settings(self):
        idx = self.camera_combo.currentData()
        resolution = self.resolution_combo.currentData()
        if self._sliders_touched:
            brightness = self.brightness_slider.value()
            contrast = self.contrast_slider.value()
            saturation = self.saturation_slider.value()
        else:
            brightness = contrast = saturation = None
        return {
            "index": idx,
            "resolution": resolution,
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
        }

    def start_preview(self, idx=None):
        settings = self._current_settings()
        if idx is not None:
            settings["index"] = idx
        self.stop_preview()
        self.worker = CameraWorker(
            settings["index"],
            resolution=settings["resolution"],
            brightness=settings["brightness"],
            contrast=settings["contrast"],
            saturation=settings["saturation"])
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.camera_error.connect(self.on_camera_error)
        self.worker.properties_ready.connect(self.on_properties_ready)
        self.worker.start()
        self.paused = False
        self.manual_corners = None
        self.frozen_frame = None
        self._drag_corner = None
        self.pause_btn.setText("暂停")
        self.pause_btn.setEnabled(True)
        self.preview_btn.setText("停止预览")
        self.capture_btn.setEnabled(True)
        self.statusBar().showMessage("预览中：%s" % self.camera_combo.currentText())

    def restart_preview(self):
        if self.worker is not None:
            self.start_preview()

    def stop_preview(self):
        if self.worker is not None:
            self.worker.frame_ready.disconnect(self.on_frame)
            self.worker.camera_error.disconnect(self.on_camera_error)
            self.worker.properties_ready.disconnect(self.on_properties_ready)
            self.worker.stop()
            self.worker = None
        self.latest_frame = None
        self.latest_corners = None
        self.paused = False
        self.frozen_frame = None
        self.manual_corners = None
        self._drag_corner = None
        self.pause_btn.setText("暂停")
        self.pause_btn.setEnabled(False)
        self.preview_btn.setText("开始预览")
        self.capture_btn.setEnabled(False)

    def on_camera_error(self, msg):
        self.stop_preview()
        QMessageBox.warning(self, "摄像头错误", msg)

    def on_properties_ready(self, props):
        ranges = props.get("ranges", {})
        actual = props.get("actual", props)
        if not self._sliders_touched:
            for name, slider in (("brightness", self.brightness_slider),
                                 ("contrast", self.contrast_slider),
                                 ("saturation", self.saturation_slider)):
                rng = ranges.get(name)
                if rng is None:
                    slider.setEnabled(False)
                    slider.setToolTip("%s：该摄像头不支持此参数" % name)
                    continue
                slider.setEnabled(True)
                slider.setRange(int(rng[0]), int(rng[1]))
                slider.setToolTip("%s：%d ~ %d" % (name, int(rng[0]), int(rng[1])))
                value = actual.get(name)
                if value is not None:
                    slider.blockSignals(True)
                    slider.setValue(int(max(rng[0], min(rng[1], value))))
                    slider.blockSignals(False)
        info = "%d x %d" % (actual["width"], actual["height"])
        self.statusBar().showMessage("预览中：%s（%s）" % (
            self.camera_combo.currentText(), info))

    # ---------- 预览画面 ----------
    def on_frame(self, payload):
        frame, corners, stable = payload
        if self.paused:
            return
        self.latest_frame = frame
        self.latest_corners = corners if corners is not None else stable
        display = frame.copy()
        box = stable if stable is not None else corners
        if box is not None:
            pts = np.int32(box.reshape(-1, 1, 2))
            cv2.polylines(display, [pts], True, (0, 255, 0), 3)
            for pt in pts:
                cv2.circle(display, tuple(pt[0]), 6, (0, 0, 255), -1)
        self._show_frame(display)

    def _show_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        self._display_h, self._display_w = h, w
        image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        pixmap = pixmap.scaled(self.preview_label.size(),
                               Qt.KeepAspectRatio,
                               Qt.SmoothTransformation)
        self.preview_label.setPixmap(pixmap)


    # ---------- 拍摄 ----------
    def capture(self):
        if self.paused:
            frame = self.frozen_frame
            corners = self.manual_corners
            manual = True
        else:
            frame = self.latest_frame
            corners = self.latest_corners
            manual = False
        if frame is None:
            return
        use_crop = self.auto_crop_check.isChecked() and corners is not None
        if use_crop:
            result = four_point_transform(frame, corners)
            prefix = "scan"
        else:
            result = frame
            prefix = "full"
        self.save_counter += 1
        ext = self.format_combo.currentText().lower()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = "%s_%s_%03d.%s" % (prefix, timestamp, self.save_counter, ext)
        path = os.path.join(self.output_dir, filename)
        cv2.imwrite(path, result)
        self.add_record(path, use_crop)
        if use_crop:
            mode = "手动裁剪" if manual else "自动裁剪"
        else:
            mode = "原画面"
        self.statusBar().showMessage("已保存：%s（%s）" % (filename, mode))


    def add_record(self, path, cropped):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            pixmap = QPixmap(180, 135)
            pixmap.fill(Qt.darkGray)
        pixmap = pixmap.scaled(QPixmap(180, 135).size(), Qt.KeepAspectRatio,
                               Qt.SmoothTransformation)
        item = QListWidgetItem(QIcon(pixmap),
                               "%s\n%s" % (os.path.basename(path),
                                          "裁剪" if cropped else "原画面"))
        item.setData(Qt.UserRole, path)
        self.record_list.addItem(item)
        self.record_list.scrollToBottom()

    def open_record(self, item):
        path = item.data(Qt.UserRole)
        if path and os.path.exists(path):
            _open_with_default_app(path)

    # ---------- 目录 ----------
    def choose_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self, "选择保存目录", self.output_dir)
        if directory:
            self.output_dir = directory

    def open_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)
        _open_with_default_app(self.output_dir)

    # ---------- 关闭 ----------
    def closeEvent(self, event):
        if self.probe is not None:
            self.probe.wait()
        self.stop_preview()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
