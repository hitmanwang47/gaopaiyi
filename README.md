# 高拍仪（文档拍摄）应用

基于 Python + PyQt5 + OpenCV 的简单高拍仪程序，参考了以下两个 GitHub 项目：

- `Names233/DocScanner`（界面交互设计）
- `andrewdcampbell/OpenCV-Document-Scanner`（角点检测与透视裁剪算法）

## 功能

- 选择摄像头（自动探测，显示为“摄像头 1、摄像头 2 …”）
- 实时预览，自动检测文档四角并画出绿色边框（多帧稳定，不会闪烁）
- 摄像头设置：分辨率（从摄像头支持列表中选取）、亮度、对比度、饱和度
- 点击“拍摄”，将文档区域透视校正后保存为 JPG / PNG
- 点“暂停”定格画面，可手动拖动四个红点微调边框后再拍摄；点“继续”恢复自动检测
- 拍摄记录列表与缩略图，双击可打开原图
- 检测不到文档时可勾掉“自动裁剪”，保存原始画面

## 运行环境

Python 3.9+，依赖库见 `requirements.txt`（numpy、opencv-python、PyQt5）：

```bash
pip install -r requirements.txt
```

## 启动

Windows：双击 `run.bat`，或手动执行：

```bash
python main.py
```

Linux/macOS：使用 `run.sh`（无需可执行权限，直接 `bash run.sh`），或手动执行：

```bash
python3 main.py
```

拍摄的图片默认保存在 `captures` 目录，可在界面中更改保存目录。

## Linux 依赖说明

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

若 PyQt5 / OpenCV 启动报缺库（如 `libGL`、`libxcb-cursor`），Debian/Ubuntu 可先安装：

```bash
sudo apt install libgl1 libegl1 libxcb-cursor0 libxkbcommon-x11-0
```

摄像头访问：若提示无法打开 `/dev/video*`，将当前用户加入 `video` 组后重新登录：

```bash
sudo usermod -aG video $USER
```

## 文档检测说明

`detector.py` 采用“阈值分割（Otsu / 自适应阈值）+ 最大连通区域 + 四边形拟合”的策略，
保留 Canny 边缘管线与最小外接矩形兜底；文档铺满画面时直接整幅识别。

预览中的绿色边框经过多帧稳定性确认：只有连续多帧位置一致的角点才会显示并保持，
短暂丢失不会闪框；点击“拍摄”时使用当前帧的检测结果裁剪，检测失败时才保存原图。
