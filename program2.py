import cv2
from deepface import DeepFace
from mtcnn import MTCNN

detector = MTCNN()
video = cv2.VideoCapture(0)

while video.isOpened():
    _, frame = video.read()
    faces = detector.detect_faces(frame)
    
    for result in faces:
        x, y, w, h = result['box']
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

        try:
            analyze = DeepFace.analyze(frame, actions=['emotion'])
            dominant_emotion = analyze[0]['dominant_emotion']
            emotion_probability = analyze[0]['emotion'][dominant_emotion]
            cv2.putText(frame, f'Emotion: {dominant_emotion} ({emotion_probability:.2f})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        except:
            cv2.putText(frame, 'No Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('video', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
