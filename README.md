🫁 TB Detector – AI-Based Tuberculosis Detection from Chest X-rays

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-orange?logo=streamlit)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-green?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A lightweight web application that uses a custom-trained **YOLOv8** deep learning model to detect **Tuberculosis (TB)** in chest X-ray images. Built using **Streamlit**, this app provides a user-friendly interface for quick and effective TB screening assistance.

🌐 Live App

👉 [Click here to try the app on Streamlit Cloud](https://tb-detector.streamlit.app)


📌 Features

- 📤 Upload chest X-ray images (JPG, JPEG, PNG)
- 🧠 Real-time AI-powered TB detection using YOLOv8
- 🎚 Adjustable confidence threshold
- 🖼 Detection results with class names and confidence scores
- 💾 Option to download the uploaded image

🛠 Tech Stack

| Component     | Description                       |
|---------------|-----------------------------------|
| Python 3.10   | Core programming language         |
| Streamlit     | Web app UI                        |
| YOLOv8        | Object detection model            |
| PyTorch       | Deep learning framework           |
| OpenCV & PIL  | Image preprocessing               |
| NumPy         | Image data handling               |


📁 Project Structure

TB_Detector/
├── app.py                 # Main Streamlit app
├── td_detect.pt           # Trained YOLOv8 model
├── requirements.txt       # Python dependencies
├── runtime.txt            # Python version (for deployment)
└── README.md              # This file

🧠 Model Details
The model (td_detect.pt) is trained using YOLOv8 on a curated dataset of chest X-rays with TB annotations. It predicts bounding boxes around suspected TB regions along with class labels and confidence scores.

🧪 Sample Prediction
![image](https://github.com/user-attachments/assets/5efd310d-0389-45e0-940f-00f19a32af00)

📦 Deployment
This app is deployed using Streamlit Cloud. To deploy your own version:

Push this repo to GitHub

Make sure the model file td_detect.pt is in the root directory

Include requirements.txt and runtime.txt

Connect your GitHub repo to Streamlit Cloud and deploy

📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

🙋‍♂️ Author
Satyanarayana Vasamsetti
📧 satyabanni11851@gmail.com
🔗 https://www.linkedin.com/in/satya-narayana-vasamsetti-0b161b2a0/

⭐ Show Your Support
If you find this project helpful, please consider giving it a ⭐ on GitHub!
