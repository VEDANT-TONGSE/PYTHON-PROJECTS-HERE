
import streamlit as st
import cv2
import numpy as np
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Self Driving Car Simulation", layout="wide")

st.title("🚗 Self-Driving Car Simulation")
st.markdown("### Real-Time Lane Detection & Steering Control")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Simulation Controls")
video_source = st.sidebar.selectbox("Video Source", ["Webcam", "Road Video"])
speed = st.sidebar.slider("Vehicle Speed", 1, 10, 5)
start = st.sidebar.button("Start Simulation")

frame_placeholder = st.empty()
info_placeholder = st.empty()

# ---------------- LANE DETECTION FUNCTIONS ----------------
def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([
        [(0, height), (img.shape[1], height), (img.shape[1], int(height*0.6)), (0, int(height*0.6))]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(img, mask)

def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    cropped = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        cropped,
        rho=2,
        theta=np.pi/180,
        threshold=100,
        minLineLength=50,
        maxLineGap=150
    )

    steering = "Straight"

    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)

            if slope > 0.5:
                steering = "Turn Right"
                cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),3)
            elif slope < -0.5:
                steering = "Turn Left"
                cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),3)

    return frame, steering

# ---------------- VIDEO CAPTURE ----------------
if start:
    if video_source == "Webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture("road.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (900, 500))
        output, steering = detect_lanes(frame)

        # Display info
        info_placeholder.markdown(
            f"""
            ### 🚦 Vehicle Status
            - **Speed:** {speed} units  
            - **Steering Direction:** **{steering}**
            """
        )

        # Show frame
        frame_placeholder.image(output, channels="BGR")

        time.sleep(0.1 / speed)

    cap.release()
    st.success("Simulation Ended")
