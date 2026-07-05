import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Setup
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
COLORS = [
    (0, 0, 255),      # Angry - red
    (0, 140, 255),    # Disgust - orange
    (0, 255, 255),    # Fear - yellow
    (0, 255, 0),      # Happy - green
    (255, 255, 0),    # Neutral - cyan
    (255, 0, 0),      # Sad - blue
    (255, 0, 255),    # Surprise - magenta
]

print("Loading model...")
model = load_model('models/v3.2_emotion_model_lighter_smoothing.keras')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam (Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Camera failed to open.")
    exit()

print("Camera on! Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=[0, -1])  # shape becomes (1, 48, 48, 1)

        preds = model.predict(roi, verbose=0)[0]
        emotion_idx = np.argmax(preds)
        emotion = EMOTIONS[emotion_idx]
        confidence = preds[emotion_idx]
        color = COLORS[emotion_idx]

        # Bounding box + label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        label = f"{emotion}: {confidence*100:.1f}%"
        cv2.rectangle(frame, (x, y-30), (x+w, y), color, -1)
        cv2.putText(frame, label, (x+5, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Confidence bars for all 7 emotions
        bar_x = x + w + 10
        for i, (emo, prob) in enumerate(zip(EMOTIONS, preds)):
            bar_y = y + i * 25
            bar_len = int(prob * 100)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 100, bar_y + 18), (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_len, bar_y + 18), COLORS[i], -1)
            cv2.putText(frame, f"{emo[:3]} {prob*100:.0f}%", (bar_x + 2, bar_y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    cv2.putText(frame, "Press Q to quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Done!")