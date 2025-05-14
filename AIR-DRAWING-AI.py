import cv2
import mediapipe as mp
import numpy as np
import pytesseract
import time
import pyttsx3

# Set path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 140)      # speech rate
engine.setProperty('volume', 1.0)    # max volume

# Set custom voice (try different voices if available)
voices = engine.getProperty('voices')
# Choose a female voice if available
for voice in voices:
    if "female" in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break  # use the first available female voice

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Video capture and canvas
cap = cv2.VideoCapture(0)
canvas = np.zeros((480, 640), dtype=np.uint8)

prev_x, prev_y = 0, 0
last_draw_time = time.time()
draw_timeout = 2  # Seconds of inactivity to recognize

recognized_text = ""
full_word = ""

five_finger_start_time = None

def count_fingers(hand_landmarks):
    landmarks = hand_landmarks.landmark
    tips_ids = [4, 8, 12, 16, 20]
    pip_ids = [2, 6, 10, 14, 18]

    fingers = []
    # Thumb (for right hand, can be improved for left)
    fingers.append(landmarks[4].x < landmarks[3].x)
    for tip, pip in zip(tips_ids[1:], pip_ids[1:]):
        fingers.append(landmarks[tip].y < landmarks[pip].y)

    return sum(fingers)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    now = time.time()

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

            x = int(hand_landmark.landmark[8].x * 640)
            y = int(hand_landmark.landmark[8].y * 480)

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y

            # Drawing on canvas
            cv2.line(canvas, (prev_x, prev_y), (x, y), 255, 10)
            last_draw_time = now
            prev_x, prev_y = x, y

            # 5-finger exit gesture
            num_fingers = count_fingers(hand_landmark)
            if num_fingers == 5:
                if five_finger_start_time is None:
                    five_finger_start_time = now
                elif now - five_finger_start_time > 1.0:
                    print("5 fingers held for 1 second. Exiting...")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()
            else:
                five_finger_start_time = None
    else:
        prev_x, prev_y = 0, 0
        five_finger_start_time = None

    # Trigger OCR if idle
    if now - last_draw_time > draw_timeout and np.sum(canvas) > 0:
        roi = canvas.copy()
        roi = cv2.resize(roi, (300, 300))
        roi = cv2.copyMakeBorder(roi, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)

        # Improve OCR accuracy
        roi_gray = cv2.cvtColor(cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2GRAY)
        roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        _, roi_thresh = cv2.threshold(roi_blur, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        custom_config = r'--oem 3 --psm 10'
        text = pytesseract.image_to_string(roi_thresh, config=custom_config).strip()

        if text and text.isalnum():
            recognized_text = text.upper()
            print(f"Recognized: {recognized_text}")
            if recognized_text == "STOP":
                engine.say("Stop detected. Goodbye!")
                engine.runAndWait()
                break
            full_word += recognized_text
            engine.say(recognized_text)
            engine.runAndWait()

        # Clear for next input
        canvas = np.zeros((480, 640), dtype=np.uint8)
        last_draw_time = now

    # Show output
    combined = cv2.addWeighted(frame, 1, cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR), 0.5, 0)
    cv2.putText(combined, f"Word: {full_word}", (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("AirDraw", combined)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break
    elif key == ord('c'):  # Clear canvas and reset word
        canvas = np.zeros((480, 640), dtype=np.uint8)
        full_word = ""

cap.release()
cv2.destroyAllWindows()
