import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture('eyeTracking/test.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            h, w, _ = frame.shape
            
            right_iris_center = (int(face_landmarks.landmark[468].x * w), int(face_landmarks.landmark[468].y * h))
            left_iris_center = (int(face_landmarks.landmark[473].x * w), int(face_landmarks.landmark[473].y * h))

            right_iris_points = [469, 470, 471, 472]
            left_iris_points = [474, 475, 476, 477]

            right_radius = np.mean([
                np.linalg.norm(np.array(right_iris_center) - np.array((int(face_landmarks.landmark[p].x * w), int(face_landmarks.landmark[p].y * h))))
                for p in right_iris_points
            ])

            left_radius = np.mean([
                np.linalg.norm(np.array(left_iris_center) - np.array((int(face_landmarks.landmark[p].x * w), int(face_landmarks.landmark[p].y * h))))
                for p in left_iris_points
            ])

            cv2.circle(frame, right_iris_center, int(right_radius), (0, 255, 0), 2)
            cv2.circle(frame, left_iris_center, int(left_radius), (0, 255, 0), 2)

    cv2.imshow("Iris Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
