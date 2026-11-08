import math
import os
import urllib.request
import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 0. 防呆下載 OpenCV Haar-Cascade 模型
# ==========================================
XML_FILENAME = "haarcascade_frontalface_default.xml"
XML_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

if not os.path.exists(XML_FILENAME):
    try:
        urllib.request.urlretrieve(XML_URL, XML_FILENAME)
    except Exception as e:
        print(f"模型下載失敗: {e}")

face_cascade = cv2.CascadeClassifier(XML_FILENAME)


# ==========================================
# 1. 核心輔助演算法 (PSNR & 直方圖)
# ==========================================
def calculate_psnr(img1, img2):
    """計算兩張影像間的 Peak Signal-to-Noise Ratio (PSNR)"""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return round(psnr, 2)


def generate_histogram(img, mode):
    """根據色彩模式繪製 0-255 動態直方圖 (相容 Matplotlib 最新版)"""
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=100)

    if mode in ["Original", "YUV (Y-Luma)", "YUV (U-Chroma)", "YUV (V-Chroma)"]:
        gray = (
            cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if len(img.shape) == 3
            else img
        )
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        ax.plot(hist, color="black", linewidth=1.5)
        ax.set_title("Intensity Histogram")
    elif mode == "RGB (Red Channel)":
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        ax.plot(hist, color="red", linewidth=1.5)
        ax.set_title("Red Channel Histogram")
    elif mode == "RGB (Green Channel)":
        hist = cv2.calcHist([img], [1], None, [256], [0, 256])
        ax.plot(hist, color="green", linewidth=1.5)
        ax.set_title("Green Channel Histogram")
    elif mode == "RGB (Blue Channel)":
        hist = cv2.calcHist([img], [2], None, [256], [0, 256])
        ax.plot(hist, color="blue", linewidth=1.5)
        ax.set_title("Blue Channel Histogram")

    ax.set_xlim([0, 256])
    ax.set_xlabel("Pixel Intensity (0-255)")
    ax.set_ylabel("Pixel Count")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # 將 Matplotlib 圖表轉為 NumPy RGB 影像陣列
    fig.canvas.draw()
    hist_img = np.asarray(fig.canvas.buffer_rgba())
    hist_img = cv2.cvtColor(hist_img, cv2.COLOR_RGBA2RGB)
    plt.close(fig)

    return hist_img


# ==========================================
# 2. 主邏輯處理函數
# ==========================================
def process_pipeline(
    image,
    color_mode,
    k00,
    k01,
    k02,
    k10,
    k11,
    k12,
    k20,
    k21,
    k22,
    normalize,
    enable_face_detection,
):
    if image is None:
        return None, None, "請先上傳圖片！", "無"

    # ------------------------------------------
    # 1. 影像載入與基礎解析
    # ------------------------------------------
    h, w, c = image.shape
    file_size_kb = round(image.nbytes / 1024, 2)
    meta_info = f"解析度: {w} × {h} | 色彩通道: {c} (RGB) | 記憶體占用: {file_size_kb} KB"

    # ------------------------------------------
    # 2. 多色彩空間轉換
    # ------------------------------------------
    processed_img = image.copy()
    if color_mode == "RGB (Red Channel)":
        processed_img[:, :, 1] = 0
        processed_img[:, :, 2] = 0
    elif color_mode == "RGB (Green Channel)":
        processed_img[:, :, 0] = 0
        processed_img[:, :, 2] = 0
    elif color_mode == "RGB (Blue Channel)":
        processed_img[:, :, 0] = 0
        processed_img[:, :, 1] = 0
    elif color_mode.startswith("YUV"):
        img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        y, u, v = cv2.split(img_yuv)
        if color_mode == "YUV (Y-Luma)":
            processed_img = cv2.cvtColor(y, cv2.COLOR_GRAY2RGB)
        elif color_mode == "YUV (U-Chroma)":
            processed_img = cv2.cvtColor(u, cv2.COLOR_GRAY2RGB)
        elif color_mode == "YUV (V-Chroma)":
            processed_img = cv2.cvtColor(v, cv2.COLOR_GRAY2RGB)

    # 渲染同步直方圖
    hist_plot = generate_histogram(processed_img, color_mode)

    # ------------------------------------------
    # 3. 自訂 3x3 卷積核空間濾波與 PSNR
    # ------------------------------------------
    kernel = np.array(
        [[k00, k01, k02], [k10, k11, k12], [k20, k21, k22]], dtype=np.float32
    )

    if normalize:
        k_sum = np.sum(kernel)
        if k_sum != 0:
            kernel /= k_sum

    filtered_img = cv2.filter2D(processed_img, -1, kernel)
    psnr_value = calculate_psnr(image, filtered_img)
    psnr_text = (
        f"{psnr_value} dB" if psnr_value != float("inf") else "∞ dB (無失真)"
    )

    # ------------------------------------------
    # 4. 人臉偵測模組
    # ------------------------------------------
    face_status = "未開啟人臉偵測"
    if enable_face_detection:
        gray_for_face = cv2.cvtColor(filtered_img, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_for_face, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        face_count = len(faces)
        for i, (x, y, fw, fh) in enumerate(faces):
            cv2.rectangle(
                filtered_img, (x, y), (x + fw, y + fh), (0, 255, 0), 3
            )
            cv2.putText(
                filtered_img,
                f"Face #{i+1}",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        face_status = f"成功偵測到 {face_count} 張人臉"

    status_summary = f"{meta_info}\nPSNR 評估: {psnr_text}\n人臉偵測結果: {face_status}"
    return filtered_img, hist_plot, status_summary, psnr_text


# ==========================================
# 3. 建立 Gradio GUI 介面
# ==========================================
with gr.Blocks(title="互動式影像處理與人臉分析工具") as demo:
    gr.Markdown("# 🎨 互動式影像處理與人臉分析工具")

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="numpy", label="1. 載入原始影像 (PNG/JPG/BMP)")

            gr.Markdown("### 🎨 2. 色彩空間切換")
            color_mode = gr.Dropdown(
                choices=[
                    "Original",
                    "RGB (Red Channel)",
                    "RGB (Green Channel)",
                    "RGB (Blue Channel)",
                    "YUV (Y-Luma)",
                    "YUV (U-Chroma)",
                    "YUV (V-Chroma)",
                ],
                value="Original",
                label="選擇色彩檢視模式",
            )

            gr.Markdown("### 🧮 3. 自訂 3x3 卷積核 (Kernel Matrix)")
            with gr.Row():
                k00 = gr.Number(value=0, label="[0,0]")
                k01 = gr.Number(value=0, label="[0,1]")
                k02 = gr.Number(value=0, label="[0,2]")
            with gr.Row():
                k10 = gr.Number(value=0, label="[1,0]")
                k11 = gr.Number(value=1, label="[1,1]")
                k12 = gr.Number(value=0, label="[1,2]")
            with gr.Row():
                k20 = gr.Number(value=0, label="[2,0]")
                k21 = gr.Number(value=0, label="[2,1]")
                k22 = gr.Number(value=0, label="[2,2]")
            normalize = gr.Checkbox(label="權重正規化 (Normalization)", value=False)

            gr.Markdown("### 🤖 4. 人臉偵測設定")
            enable_face_detection = gr.Checkbox(
                label="開啟 Haar-Cascade 人臉偵測", value=True
            )
            btn_run = gr.Button("🚀 執行處理與分析", variant="primary")

        with gr.Column(scale=1):
            output_img = gr.Image(label="處理後影像 (濾鏡/人臉標示)")
            hist_output = gr.Image(label="同步直方圖分析 (Histogram)")
            status_text = gr.Textbox(label="中繼資訊與評估報告", lines=5)

    inputs = [
        input_img,
        color_mode,
        k00,
        k01,
        k02,
        k10,
        k11,
        k12,
        k20,
        k21,
        k22,
        normalize,
        enable_face_detection,
    ]
    outputs = [output_img, hist_output, status_text]

    btn_run.click(fn=process_pipeline, inputs=inputs, outputs=outputs)
    input_img.change(fn=process_pipeline, inputs=inputs, outputs=outputs)
    color_mode.change(fn=process_pipeline, inputs=inputs, outputs=outputs)

# 啟動時自動彈出瀏覽器
if __name__ == "__main__":
    demo.launch(inbrowser=True, share=False)