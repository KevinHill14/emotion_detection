import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Setup emotions and colours
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
COLOURS = [
    (0, 0, 255),      # Angry - red
    (0, 140, 255),    # Disgust - orange
    (0, 255, 255),    # Fear - yellow
    (0, 255, 0),      # Happy - green
    (255, 255, 0),    # Neutral - cyan
    (255, 0, 0),      # Sad - blue
    (255, 0, 255),    # Surprise - magenta
]

# Load pretrained model
print("Loading model...")
model = load_model('main_model/v3.2_emotion_model_lighter_smoothing.keras')

# Initialize face detector with Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)
time.sleep(2)

if not cap.isOpened():
    print("Camera failed to open.")
    exit()
print("Camera on! Press ESC to quit.")

while True:
    ret, frame = cap.read()
    # If failed to return a frame, exit
    if not ret:
        print("Failed to return frame")
        break
    
    # Convert frame to grayscale and detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)

    # Safeguard to keep only the largest face
    if len(faces) > 0:
        faces = [max(faces, key=lambda f: f[2] * f[3])]
    else:
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # If nothing is detected, display a message
    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    for (x, y, w, h) in faces:
        # Add a 15% padding around detected faces to maintain facial features
        pad = int(0.15 * w)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(gray.shape[1], x + w + pad)
        y2 = min(gray.shape[0], y + h + pad)

        # Extract ROI and process for the model
        roi = gray[y1:y2, x1:x2]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=[0, -1])

        # Pass image to model and find the highest probability
        preds = model.predict(roi, verbose=0)[0]
        emotion_idx = np.argmax(preds)

        # Set the prediction to corresponding emotion
        emotion = EMOTIONS[emotion_idx]
        confidence = preds[emotion_idx]
        color = COLOURS[emotion_idx]

        # Draw bounding box and label around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        label = f"{emotion}: {confidence*100:.1f}%"
        cv2.rectangle(frame, (x, y-30), (x+w, y), color, -1)
        cv2.putText(frame, label, (x+5, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw confidence bars for emotions
        bar_x = x + w + 10
        for i, (emo, prob) in enumerate(zip(EMOTIONS, preds)):
            bar_y = y + i * 25
            bar_len = int(prob * 100)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 100, bar_y + 18), (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_len, bar_y + 18), COLOURS[i], -1)
            cv2.putText(frame, f"{emo[:3]} {prob*100:.0f}%", (bar_x + 2, bar_y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
    # Display the window
    cv2.imshow('Emotion Detector', frame)

    # Break the loop if ESC is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Exitting!")

# Terminal input - python src/webcam_detect.py
