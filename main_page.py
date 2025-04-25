import gradio as gr
from backend_models import process_segmentation, classify_yolo, classify_efficientnet
from ultralytics import YOLO


def detect_disease(image):
    segmented_image, disease_info = process_segmentation(image)
    return segmented_image, disease_info


def classify_leaf(image, model_choice):
    if model_choice == "YOLO":
        leaf_class, health_status = classify_yolo(image)
    elif model_choice == "EfficientNet":
        leaf_class, health_status = classify_efficientnet(image)
    else:
        leaf_class, health_status = "Unknown", "Unknown"
    return leaf_class, health_status


with gr.Blocks() as demo:
    gr.Markdown("## Plant Disease Detection and Leaf Classification")

    # Row for image upload and immediate display of the original image.
    with gr.Row():
        image_input = gr.Image(label="Upload Image", type="pil")
        original_display = gr.Image(label="Original Image")

    # Display the uploaded image.
    image_input.change(lambda img: img, image_input, original_display)

    # Create tabs for segmentation and classification.
    with gr.Tabs():
        with gr.Tab("Disease Segmentation"):
            seg_button = gr.Button("Detect Disease")
            seg_image_output = gr.Image(label="Segmentation Result")
            seg_text_output = gr.Textbox(label="Disease Info")
            seg_button.click(fn=detect_disease, inputs=image_input, outputs=[seg_image_output, seg_text_output])

        with gr.Tab("Leaf Classification"):
            model_selector = gr.Radio(choices=["YOLO", "EfficientNet"],
                                      label="Select Classification Model",
                                      value="EfficientNet")
            class_button = gr.Button("Classify Leaf")
            class_text_output = gr.Textbox(label="Leaf Class")
            health_text_output = gr.Textbox(label="Health Status")
            class_button.click(fn=classify_leaf, inputs=[image_input, model_selector],
                               outputs=[class_text_output, health_text_output])

    demo.launch()
