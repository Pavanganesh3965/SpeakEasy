# SpeakEasy
Speakeasy is a deep learning-based lip-reading model designed to convert lip movements from videos into text. The model utilizes Conv3D layers for spatial-temporal feature extraction and Bidirectional LSTMs for modeling the sequential nature of lip movements. Trained on the GRID dataset, it aligns lip movements with phonemes to accurately predict spoken words without audio. The project aims to improve accessibility for the hearing-impaired, enhance communication in noisy environments, and facilitate human-computer interactions. With an impressive 93% accuracy, Speakeasy demonstrates the power of AI in understanding visual speech and is a step forward towards real-time lip-reading applications.

## Problem Statement
This project aims to develop an end-to-end machine learning solution for accurate word detection from video recordings of individuals speaking, using deep learning techniques like LSTM and Neural Networks to analyze visual cues. The goal is to enhance communication accessibility, particularly in noisy environments where traditional speech recognition struggles. For instance, in crowded spaces, airports, or emergency situations, lip reading offers a reliable, silent communication method. Additionally, this technology can be used in CCTV systems to improve security by reading lips in surveillance footage, and it plays a vital role for the hearing-impaired community, enabling better understanding of speech in environments where audio may be unclear or unavailable. By leveraging lip-reading technology, this system provides a versatile solution for clear communication without relying on sound.

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.x-blue?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

## Features
1. **Accurate Lip-Reading:** Utilizes deep learning techniques, including LSTM (Long Short-Term Memory) networks and 3D Convolutional Neural Networks (Conv3D), to accurately detect words from lip movements in video footage.

2. **Real-Time Performance:** Capable of processing video frames in real-time to predict spoken words from lip movements, providing immediate output.

3. **Enhanced Accessibility:** Helps hearing-impaired individuals by enabling better communication through visual cues without relying on audio.

4. **Works in Noisy Environments:** The system provides an alternative to traditional speech recognition, making it effective in environments where audio input is noisy or unavailable (e.g., crowded spaces, emergency situations, or factories).

5. **Multimodal Learning:** The model integrates both spatial and temporal features from videos, effectively capturing both the shape and movement of the lips over time.

6. **High Accuracy:** Achieves 93% accuracy in word detection from lip movements, demonstrating robust performance in real-world applications.

7. **CCTV Integration:** Can be applied to CCTV systems for surveillance and security purposes, where lip reading can help extract information from video footage when audio is not available.

8. **Customizable for Multiple Languages:** The model can be trained on datasets in different languages, enabling multilingual lip reading capabilities.

### Prerequisites
- Python (3.x)
- TensorFlow (2.x)
- Keras
- OpenCV
- NumPy
- Matplotlib
- FFmpeg
 
- Jupyter Notebook (optional)
- GRID Dataset
- Git

## Folder Structure
```bash
SI-GuidedProject-601004-1700586832/
│
├── checkpoints.zip
├── LipNet.ipynb
├── README.md
├── .git
├── .gitignore
├── .venv
│
├── app/
│   ├── modelutil.py
│   ├── streamlitapp.py
│   ├── test_video.mp4
│   ├── utils.py
│   ├── animation.gif
│   ├── Lip_Reading_-__Environment.yml
│   ├── __pycache__/
│
├── data/
│   ├── alignments/
│   ├── s1/
│
├── models/
│   ├── checkpoint.data-00000-of-00001
│   ├── checkpoint.h5
│   ├── checkpoint.index
│   ├── checkpoint.keras
│   ├── checkpoint
│   ├── __MACOSX/
 ```

# ScreenShots:

