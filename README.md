# character-digit-recognizer
# Smart Handwritten Character Recognition App

An interactive, high-accuracy web application that recognizes hand-drawn **digits (0-9)**, **uppercase letters (A-Z)**, and **lowercase letters (a-z)** in real-time. 

Built using a deep **Convolutional Neural Network (CNN)** trained on the massive EMNIST ByClass dataset (62 unique classes), this project showcases a seamless integration of a deep learning model with an intuitive Python web interface via Streamlit.

---

## 🚀 Live Demo
🔗 **[Click here to view the live web application](https://character-digit-recognizer-abhishek.streamlit.app/)** 
---

## ✨ Features
- **62-Class Recognition Engine:** Seamlessly classifies digits, uppercase, and lowercase English letters all at once.
- **Real-Time Canvas Drawing:** An interactive sketch box that allows users to write naturally using a mouse or touchscreen.
- **Advanced Preprocessing Pipeline:** Features automated character bounding-box cropping and padding to auto-center inputs, replicating the exact structure of the EMNIST training dataset for maximized accuracy.
- **Cross-Platform Compatibility:** Fully optimized using modern Keras 3 / TensorFlow 2.16+ specifications.

---

## 🛠️ Tech Stack & Tools
- **Model Training:** Python, Google Colab (T4 Cloud GPU), TensorFlow/Keras
- **Dataset:** EMNIST ByClass (via TensorFlow Datasets)
- **Frontend UI:** Streamlit, Streamlit Drawable Canvas
- **Image Processing:** PIL (Pillow), NumPy

---

## 🧠 Model Architecture & Training
To map the complex differences between 62 distinct character variants (such as distinguishing a lowercase `o` from a zero `0`), the underlying network utilizes a deep convolutional design:
1. **Conv2D Block 1:** 32 filters (3x3), Batch Normalization, MaxPooling (2x2), and 25x% Dropout.
2. **Conv2D Block 2:** 64 filters (3x3), Batch Normalization, MaxPooling (2x2), and 25% Dropout.
3. **Fully Connected Layer:** Flattening layer leading into a Dense layer of 512 neurons, Batch Normalization, and a heavy 50% Dropout layer to prevent overfitting.
4. **Output Layer:** Dense layer with 62 nodes utilizing a Softmax activation function to convert raw logs into precision probability matrices.

The model was optimized using the **Adam optimizer** (`learning_rate=0.001`) and evaluated via **Sparse Categorical Crossentropy** over 5 rigorous cloud training epochs.

---

## 💻 Local Installation & Setup

If you want to run this application locally on your laptop, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd handwritten-character-recognizer
   ```

2. **Set up a compatible environment and install dependencies:**
   Make sure you are using Python 3.11 or Python 3.12 for maximum framework stability.
   ```bash
   python -m pip install --upgrade pip
   python -m pip install tensorflow>=2.16.1 keras>=3.0.0 numpy streamlit streamlit-drawable-canvas pillow
   ```

3. **Launch the Streamlit web dashboard:**
   ```bash
   python -m streamlit run app.py
   ```
   The app will automatically pop open on your local web profile at `http://localhost:8501`.

---

## 📁 File Structure
```text
├── app.py                     # Main Streamlit web application interface
├── emnist_model.h5            # Converted standalone high-accuracy Keras 3 model weights
├── requirements.txt           # Cloud deployment configuration dependency file
└── README.md                  # Comprehensive repository documentation manual
```

---

## 👤 Author
- **Abhishek Jadhav** - *Initial Developer* - (https://github.com/Abhishek-45-AJ)
