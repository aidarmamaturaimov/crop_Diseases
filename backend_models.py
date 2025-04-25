import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
import gradio as gr

# ──────────────────────────────────────────────────────────────────────────────
# 1) Load your models
# ──────────────────────────────────────────────────────────────────────────────

SEG_MODEL_PATH     = "/Users/aidarmamaturaimov/Downloads/runs/segment/train5/weights/best.pt"
assert os.path.exists(SEG_MODEL_PATH)
seg_model          = YOLO(SEG_MODEL_PATH)

YOLO_CLS_PATH      = "/Users/aidarmamaturaimov/PycharmProjects/PythonProject1/runs/classify/train6/weights/last.pt"
assert os.path.exists(YOLO_CLS_PATH)
yoloClassification = YOLO(YOLO_CLS_PATH)

EF_MODEL_PATH      = "/Users/aidarmamaturaimov/Downloads/models/FIP_model.keras"
assert os.path.exists(EF_MODEL_PATH)
ef_model           = load_model(EF_MODEL_PATH)

CLASS_MAPPING = {
    0: "Pepper, bell, Bacterial spot",  1: "Pepper, bell, healthy",
    2: "Potato Early blight",            3: "Potato Late blight",
    4: "Potato healthy",                 5: "Tomato Bacterial spot",
    6: "Tomato Early blight",            7: "Tomato Late blight",
    8: "Tomato Leaf Mold",               9: "Tomato Septoria leaf spot",
    10: "Spider mites Two-spotted spider mite",
    11: "Tomato Target Spot",
    12: "Tomato Yellow Leaf Curl Virus",
    13: "Tomato mosaic virus",
    14: "Tomato healthy"
}

# ──────────────────────────────────────────────────────────────────────────────
# 2) Combined detect & classify
# ──────────────────────────────────────────────────────────────────────────────

def detect_and_classify(img: Image.Image):
    # prepare inputs
    im_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    im_256 = cv2.resize(im_bgr, (256, 256))
    im_640 = cv2.resize(im_bgr, (640, 640))

    # 1) Segmentation overlay
    seg_res  = seg_model.predict(source=im_256, conf=0.5, imgsz=256)[0]
    has_mask = bool(seg_res.masks and len(seg_res.masks.data))
    if has_mask:
        m = seg_res.masks.data.cpu().numpy()[0]
        if m.shape != im_256.shape[:2]:
            m = cv2.resize(m, (256,256), interpolation=cv2.INTER_NEAREST)
        ov = im_256.copy(); ov[m>0.5] = [255,0,0]
        seg_img = cv2.addWeighted(im_256, 0.5, ov, 0.5, 0)
    else:
        seg_img = im_256.copy()
    seg_out = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)

    # 2) YOLO Classification (640×640)
    cls2    = yoloClassification.predict(source=im_640, imgsz=640)[0].probs
    idx2    = int(cls2.top1)
    c2      = float(cls2.top1conf)
    yolo_lbl   = yoloClassification.names[idx2]
    yolo_conf  = f"{c2*100:.1f}%"
    yolo_health = "Healthy" if "healthy" in yolo_lbl.lower() else "Sick"

    # 3) EfficientNet Classification
    if has_mask:
        p128 = img.resize((128,128))
        arr  = keras_image.img_to_array(p128)[None]/255.0
        pr   = ef_model.predict(arr)
        idx3 = int(np.argmax(pr, axis=1)[0])
        e_lbl  = CLASS_MAPPING[idx3]
        e_conf = f"{float(np.max(pr))*100:.1f}%"
        health = "Healthy" if "healthy" in e_lbl.lower() else "Sick"
    else:
        e_lbl, e_conf, health = "Healthy", "100.0%", "Healthy"

    return seg_out, yolo_lbl, yolo_conf, yolo_health, e_lbl, e_conf, health

# ──────────────────────────────────────────────────────────────────────────────
# 3) Gradio interface
# ──────────────────────────────────────────────────────────────────────────────

with gr.Blocks() as demo:
    gr.Markdown("## 🌿 Plant Leaf Disease Segmentation & Classification")

    # Upload + segmentation preview
    with gr.Row():
        inp     = gr.Image(type="pil", label="Original Leaf Image")
        seg_img = gr.Image(label="Segmentation Image")

    # Single button
    btn = gr.Button(" Detect & Classify Disease", variant="primary")

    # YOLO vs EfficientNet columns with divider
    with gr.Row():
        with gr.Column():
            gr.Markdown("###  YOLO Classification")
            yolo_lbl     = gr.Textbox(label="Class", interactive=False)
            yolo_conf    = gr.Textbox(label="Confidence", interactive=False)
            yolo_health  = gr.Textbox(label="Health Status", interactive=False)

        gr.HTML("<div style='width:2px; background:#444; margin:0 16px;'></div>")

        with gr.Column():
            gr.Markdown("### EfficientNet Classification")
            eff_lbl        = gr.Textbox(label="Class", interactive=False)
            eff_conf       = gr.Textbox(label="Confidence", interactive=False)
            health_status  = gr.Textbox(label="Health Status", interactive=False)

    btn.click(
        fn=detect_and_classify,
        inputs=[inp],
        outputs=[seg_img, yolo_lbl, yolo_conf, yolo_health, eff_lbl, eff_conf, health_status]
    )

    demo.launch()
