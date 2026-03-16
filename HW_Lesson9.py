import cv2
import os

net = cv2.dnn.readNetFromCaffe('data/mobilenet/mobilenet_deploy.prototxt', 'data/mobilenet/mobilenet.caffemodel')
classes = []

with open('data/mobilenet/synset.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split(' ', 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)

image_folder = 'images/MobileNet'
files = os.listdir(image_folder)

class_counts = {}

for file in files:
    path = image_folder + "/" + file
    image = cv2.imread(path)
    
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (224, 224)),
        1.0 / 127.5,
        (224, 224),
        (127.5, 127.5, 127.5)
    )

    net.setInput(blob)
    preds = net.forward()

    index = preds[0].argmax()

    label = classes[index] if index < len(classes) else "unknown"
    conf = float(preds[0][index]) * 100

    print("Файл:", file)
    print("Клас:", label)
    print("Впевненість:", round(conf, 2), "%\n")
    
    if label in class_counts:
        class_counts[label] += 1
    else:
        class_counts[label] = 1

print("Клас | Кількість")
for label in class_counts:
    print(label, "|", class_counts[label])
