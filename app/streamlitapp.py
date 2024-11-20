import streamlit as st
import os
import time
import imageio

import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

st.set_page_config(layout='wide', page_title="SpeakEasy - Lip Reading", page_icon="🎤")

# Sidebar with project details
with st.sidebar:
    st.image('https://i.ibb.co/x22dPZG/Speak-Easy-Documentation-removebg-preview.png', width=250)
    st.title(':red[SpeakEasy]')
    st.info(
        'The objective of this project is to develop an end-to-end machine learning solution '
        'to detect words from a video of a person speaking.'
    )
    if st.button('Meet the Team 👥'):
        st.subheader('Team Speak Easy')
        st.write("1. MUSTURU PAVAN GANESH")
        st.write("2. SURAM SAI YASWANTH REDDY")
        st.write("3. MUNAMALA VENKATA SAI PREETHAM")
        st.write("**Guide:** Dr. ARIVARASU M")

# Main title
st.title(':red[SPEAK EASY - Lip Reading using Deep Learning] 🎥🧠')
st.divider()

# Video selection
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('🎬 Select a Video for Analysis:', options)

col1, col2 = st.columns(2)

if options:
    # Column 1: Displaying the video
    with col1:
        st.info('🎥 The video below displays the converted video in mp4 format.')
        with st.spinner('Rendering the video for analysis... 🔄'):
            file_path = os.path.join('..', 'data', 's1', selected_video)
            os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')
            video = open('test_video.mp4', 'rb')
            video_bytes = video.read()
            st.video(video_bytes)

    st.write("\n")

    # Column 2: Machine Learning visualization and predictions
    with col2:
        st.info('👁️ This is all the machine learning model sees when making a prediction.')
        video_data, annotations = load_data(tf.convert_to_tensor(file_path))
        st.image('animation.gif', width=400, caption="Model's Perspective")

        st.info('📜 This is the raw output of the machine learning model as tokens.')

        # Processing predictions with progress animations
        with st.spinner('🔍 Performing Lip Reading...'):
            model = load_model()
            yhat = model.predict(tf.expand_dims(video_data, axis=0))
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
            st.text(f"Raw Tokens: {decoder}")
        st.info('📝 Decoding the raw tokens into meaningful words.')
        progress_message = st.empty()
        progress_bar = st.progress(0)

        # Simulating decoding process
        for i in range(1, 101, 10):  # Smooth updates
            progress_message.text(f"Decoding... {i}%")
            time.sleep(0.1)
            progress_bar.progress(i)

        # Ensure progress bar reaches 100% after the loop
        progress_message.text("Decoding... 100%")
        progress_bar.progress(100)

        # Display the decoded text
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text_area("📖 Decoded Text:", converted_prediction, height=100)

st.success('🎉 Speech Successfully Interpreted! Great Work!')

