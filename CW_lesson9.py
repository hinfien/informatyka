import cv2

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
    
image = cv2.imread('images/in/images.jpg')

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

print("Клас:", label)
print("Немовірність:", round(conf, 2), "%")

text = label + " (" + str(round(conf, 2)) + "%)"
cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imshow('Image', image)
cv2.waitKey(0)