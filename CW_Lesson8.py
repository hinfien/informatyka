import cv2
import numpy as np
import os

input_folder = 'images/in'
output_folder = 'images/out'

formats = ('.jpg', '.jpeg', '.png', '.webp', '.tiff', '.tif')

os.makedirs(output_folder, exist_ok=True)

face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')

face_net = cv2.dnn.readNetFromCaffe('data/dnn/deploy.prototxt', 'data/dnn/res10_300x300_ssd_iter_140000.caffemodel')

files = sorted(os.listdir(input_folder))

for file in files:
    if not file.lower().endswith(formats):
        continue

    path = os.path.join(input_folder, file)
    
    frame = cv2.imread(path)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # __________DNN__________
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")
            
            x, y = max(0, x), max(0, y)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

            # ой очі блакитні очі
            roi_gray = gray[y:y2, x:x2]
            roi_color = frame[y:y2, x:x2]
            eyes = eye_cascade.detectMultiScale(
                roi_gray, 
                scaleFactor=1.1, 
                minNeighbors=2,
                minSize=(15, 15)
            )
            
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

    output_path = os.path.join(output_folder, file)
    cv2.imwrite(output_path, frame)

    cv2.imshow('Result', frame)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()