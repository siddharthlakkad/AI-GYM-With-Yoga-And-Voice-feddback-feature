import cv2
import mediapipe as mp
import numpy as np
import time
import speech_recognition as sr
import pyttsx3
import threading
import queue
import streamlit as st

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed percent (can go over 100)
engine.setProperty('volume', 0.9)  # Volume 0-1

# Create a queue for speech tasks
speech_queue = queue.Queue()

# Mediapipe solutions
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Recognize exercise from speech
def recognize_exercise():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        st.write("Please say something...")
        audio = r.listen(source)
        st.write("Recognizing now...")
        try:
            exercise = r.recognize_google(audio)
            st.write(f"Exercise recognized: {exercise}")
            return exercise.lower()
        except Exception as e:
            st.write(f"Error with the recognition service: {e}")
        return None

# Function to handle text-to-speech feedback in a separate thread
def speech_worker():
    while True:
        feedback = speech_queue.get()
        if feedback is None:
            break
        st.write(f"Speaking feedback: {feedback}")
        engine.say(feedback)
        engine.runAndWait()
        speech_queue.task_done()

# Start the speech worker thread
speech_thread = threading.Thread(target=speech_worker)
speech_thread.daemon = True
speech_thread.start()

# Function to handle feedback for both text and voice


def process_dumbbell():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    counter = 0
    stage = None
    feedback = ''
    feedback_text = ''
    last_feedback = ''
    start_time = None

    # Placeholder for video feed
    frame_placeholder = st.empty()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.6) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # ReColor Image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    angle_left = calculate_angle(shoulder_left, elbow_left, wrist_left)
                    angle_right = calculate_angle(shoulder_right, elbow_right, wrist_right)

                    cv2.putText(image, str(angle_left), tuple(np.multiply(elbow_left, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, str(angle_right), tuple(np.multiply(elbow_right, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                    # Dumbbell Curl logic
                    if angle_left > 160 and angle_right > 160:
                        if stage == 'up':
                            counter += 1
                            feedback = 'Good job!'
                        stage = 'down'
                        start_time = time.time()
                    if angle_left < 30 and angle_right < 30:
                        stage = 'up'
                        feedback = 'Curl up'
                        if start_time:
                            elapsed_time = time.time() - start_time
                            if elapsed_time < 3:
                                feedback = 'Slow down!'
                            start_time = None

                    # Additional feedback for form
                    if angle_left < 160 and angle_left > 100 and angle_right < 160 and angle_right > 100:
                        feedback = 'Extend your arms fully'
                    if abs(elbow_left[0] - shoulder_left[0]) > 0.1 or abs(elbow_right[0] - shoulder_right[0]) > 0.1:
                        feedback = 'Keep elbows close to body'

                    if feedback and feedback != last_feedback:
                        st.write(f"Feedback for dumbbell curls: {feedback}")
                        speech_queue.put(feedback)
                        feedback_text = feedback
                        last_feedback = feedback
                        feedback = ''

            except Exception as e:
                st.write(f"Error processing frame: {e}")

            cv2.rectangle(image, (0, 0), (1200, 80), (1000, 700, 75), -1)
            cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE', (65, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'FEEDBACK:', (350, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, feedback_text, (350, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'DUMBBELL CURLS', (850, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=4, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=4, circle_radius=4))

            # Update the frame placeholder
            frame_placeholder.image(image, channels='BGR')

    cap.release()

if __name__ == "__main__":
    st.title("Personal Gym Assistant")
    st.write("Select an exercise to start the recognition and feedback.")
    exercise = recognize_exercise()
    if exercise:
        if "dumbbell" in exercise:
            speech_queue.put("Starting dumbbell curls exercise. Get ready")
            process_dumbbell()
    else:
        st.write("Exercise not recognized. Please try again.")

    # Signal the speech thread to exit
    speech_queue.put(None)
    speech_thread.join()
