import cv2
import time
import numpy as np

haar = cv2.data.haarcascades
face_cascade  = cv2.CascadeClassifier(haar + 'haarcascade_frontalface_default.xml')
eye_cascade   = cv2.CascadeClassifier(haar + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(haar + 'haarcascade_smile.xml')
if face_cascade.empty() or eye_cascade.empty() or smile_cascade.empty():
    raise RuntimeError("Haar cascades failed to load. Check OpenCV installation.")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Camera not found. Try a different index.")

COL_FACE  = (128, 0, 128)
COL_EYES  = (255, 255, 255)
COL_SMILE = (0, 0, 255)
COL_TEXT  = (255, 255, 255)

t_prev = time.time()
fps_ema = 0.0
alpha = 0.15

print("Running... Press ESC to quit")

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    t_now = time.time()
    dt = max(t_now - t_prev, 1e-6)
    t_prev = t_now
    inst_fps = 1.0 / dt
    fps_ema = inst_fps if fps_ema == 0.0 else (alpha * inst_fps + (1 - alpha) * fps_ema)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.15, minNeighbors=6, minSize=(110, 110)
    )

    msg = ""
    face_roi = None

    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x + w, y + h), COL_FACE, 2)

        roi_g = gray[y:y + h, x:x + w]
        roi_c = frame[y:y + h, x:x + w]
        face_roi = roi_c.copy()

        eyes = eye_cascade.detectMultiScale(
            roi_g, scaleFactor=1.08, minNeighbors=12, minSize=(28, 28)
        )
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_c, (ex, ey), (ex + ew, ey + eh), COL_EYES, 2)


        smiles = smile_cascade.detectMultiScale(
            roi_g, scaleFactor=1.25, minNeighbors=24, minSize=(45, 45)
        )
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_c, (sx, sy), (sx + sw, sy + sh), COL_SMILE, 2)

        if len(eyes) < 2:
            msg = "Открой глаза"
        elif len(smiles) == 0:
            msg = "Улыбнись"
        else:
            msg = ""

        break

    cv2.putText(frame, f"FPS: {fps_ema:.1f}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps_ema:.1f}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_TEXT, 2, cv2.LINE_AA)

    if msg:
        cv2.putText(frame, msg, (14, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, msg, (14, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.95, COL_TEXT, 2, cv2.LINE_AA)

    cv2.imshow("Haar Face/Eyes/Smile - Robot UI", frame)
    if face_roi is not None:
        cv2.imshow("Your Face (ROI)", face_roi)

    if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()