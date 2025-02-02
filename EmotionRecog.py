import cv2
import numpy as np
from deepface import DeepFace

# access cam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # grayscale for acc
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Convert back to RGB for DeepFace
        rgb_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)  
        
        # Analyze emotion
        analysis = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=True)
        emotion = analysis[0]['dominant_emotion']

    # if no face is in frame
    except Exception as e:
        emotion = "No face detected"

    # Display detected emotion
    cv2.putText(frame, f'Emotion: {emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Emotion Recognition", frame)

    # escape program
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
