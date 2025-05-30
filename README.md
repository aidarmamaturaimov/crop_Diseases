# Crop Disease Detection using AI

This project is part of my final year work (FYP) for my BSc in Computer Science (Artificial Intelligence) at Brunel University London. The goal is to detect and classify crop diseases using advanced computer vision techniques, providing farmers with early warnings and suggestions for treatment.

---

## 🧠 Project Overview

The project combines image classification and segmentation models to identify plant leaf diseases. It uses both EfficientNet and YOLOv8 architectures and is deployed through a simple web app interface using Gradio.

### Key Features:
- **EfficientNet model** for image classification (FYP_final.ipynb)
- **YOLOv8 model** for object detection and segmentation (YOLO_FIP.ipynb, Leaves_class.py)
- **Gradio UI** for user interaction (main_page.py)
- **Plotly visualizations** to display results (plotly.py)
- **Backend integration** for model handling (backend_models.py)

---

## 📁 Project Structure

```
├── FYP_final.ipynb         # EfficientNet classification notebook
├── YOLO_FIP.ipynb          # YOLOv8 segmentation notebook
├── Leaves_class.py         # YOLOv8 classification model script
├── backend_models.py       # Gradio backend integration
├── main_page.py            # Gradio web interface
├── plotly.py               # Plotly visualizations for results
├── README.md               # Project documentation
```

---

## 🚀 How to Run

1. Clone this repository:
```bash
git clone https://github.com/aidarmamaturaimov/crop_Diseases.git
cd crop_Diseases
```
2. Install the required libraries:
```bash
pip install -r requirements.txt
```
3. Run the app:
```bash
python main_page.py
```

---

## 🔧 Technologies Used
- Python
- TensorFlow & Keras
- YOLOv8 (Ultralytics)
- EfficientNet
- Gradio
- Plotly

---

## 🎓 Author
**Aidar Mamaturaimov**  
Final Year BSc Computer Science (AI), Brunel University London  
Email: mamaturaimovaydar@gmail.com  
GitHub: [aidarmamaturaimov](https://github.com/aidarmamaturaimov)

---

## 📌 Future Work
- Improve segmentation accuracy with more labeled data
- Add multilingual support to Gradio interface
- Deploy as a web service for real-time access

Feel free to fork or suggest improvements!
