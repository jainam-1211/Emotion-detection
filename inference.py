import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model 

model  = load_model("model.h5")
label = np.load("labels.npy")

emotions = ["happy", "fear", "excitement", "angry", "neutral", "sad", "sleeping", "surprised", "thoughtful"]

holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    lst = []

    _, frm = cap.read()

    frm = cv2.flip(frm, 1)


    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for _ in range(42):
                lst.append(0.0)

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for _ in range(42):
                lst.append(0.0)

        lst = np.array(lst).reshape(1, -1)

        predictions = model.predict(lst)[0]
        max_index = np.argmax(predictions)
        max_emotion = label[max_index]

        # cv2.putText(frm, max_emotion, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

        # Display all emotions with percentages as horizontal bars
        for idx, emotion in enumerate(emotions):
            bar_length = int(predictions[idx] * 200)  # Scale the bar length
            cv2.putText(frm, f"{emotion}: {predictions[idx]*100:.2f}%", (10, 80 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(frm, (150, 70 + idx * 30), (150 + bar_length, 90 + idx * 30), (0, 255, 0), -1)

    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    cv2.imshow("window", frm)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break

# import cv2 
# import numpy as np 
# import mediapipe as mp 
# from keras.models import load_model 
# import matplotlib.pyplot as plt

# model  = load_model("model.h5")
# label = np.load("labels.npy")

# emotions = ["happy", "fear", "excitement", "angry", "neutral", "sad", "sleeping", "surprised", "thoughtful"]

# holistic = mp.solutions.holistic
# hands = mp.solutions.hands
# holis = holistic.Holistic()
# drawing = mp.solutions.drawing_utils

# cap = cv2.VideoCapture(0)

# plt.ion()  # Turn on interactive mode for matplotlib

# while True:
#     lst = []

#     _, frm = cap.read()

#     frm = cv2.flip(frm, 1)

#     res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

#     if res.face_landmarks:
#         for i in res.face_landmarks.landmark:
#             lst.append(i.x - res.face_landmarks.landmark[1].x)
#             lst.append(i.y - res.face_landmarks.landmark[1].y)

#         if res.left_hand_landmarks:
#             for i in res.left_hand_landmarks.landmark:
#                 lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
#                 lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
#         else:
#             for _ in range(42):
#                 lst.append(0.0)

#         if res.right_hand_landmarks:
#             for i in res.right_hand_landmarks.landmark:
#                 lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
#                 lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
#         else:
#             for _ in range(42):
#                 lst.append(0.0)

#         lst = np.array(lst).reshape(1, -1)

#         predictions = model.predict(lst)[0]

#         # Calculate emotion percentages
#         emotion_percentages = {emotion: value * 100 for emotion, value in zip(emotions, predictions)}

#         # Clear previous plot
#         plt.clf()

#         # Plotting the circular wheel
#         plt.pie(list(emotion_percentages.values()), labels=list(emotion_percentages.keys()), startangle=90, counterclock=False, autopct='%1.1f%%')
#         plt.title('Emotion Wheel')

#         # Update plot
#         plt.pause(0.001)

#     drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
#     drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
#     drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

#     cv2.imshow("window", frm)

#     if cv2.waitKey(1) == 27:
#         cv2.destroyAllWindows()
#         cap.release()
#         break

