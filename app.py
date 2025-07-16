

import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st


st.set_page_config(layout="wide")
st.image('CHATBOT TITLE.png')

col1, col2 = st.columns([3, 2])

with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Answer")
    output_text_area = st.subheader("")

genai.configure(api_key="AIzaSyBuLdRCRjoUuKzGug2LXMpwsc4oKsYTmok")
model = genai.GenerativeModel('gemini-1.5-flash')


cap = cv2.VideoCapture(0)  # '0' for built-in camera, use '1' for external camera if needed
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList

    else:
        return None

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None

    if fingers == [0, 1, 0, 0, 0]:  # Index finger up for drawing
        current_pos = lmList[8][0:2]
        if prev_pos is None: 
            prev_pos = current_pos
        cv2.line(canvas, tuple(prev_pos), tuple(current_pos), (255, 255, 0), 15)
        prev_pos = current_pos
    else:
        prev_pos = None
    if fingers == [1, 1, 1, 1, 1]:
        canvas = np.zeros_like(canvas)
        prev_pos = None
    return prev_pos, canvas

def sendToAI(model, canvas, fingers):
    if fingers == [1, 0, 0, 0, 0]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve and Explain the math problem.", pil_image])
        return response.text
    return ""

prev_pos = None
canvas = None
output_text = ""

while run:
    success, img = cap.read()

    if not success:
        st.write("Failed to capture image from webcam.")
        break
    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)
    info = getHandInfo(img)

    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_text = sendToAI(model, canvas, fingers)

    # Combine the canvas and the webcam image
    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")

    # Update the AI-generated text if available
    if output_text:
        output_text_area.text(output_text)