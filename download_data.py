import urllib.request
import os

files = {
    'data/mobilenet/mobilenet_deploy.prototxt': 'https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_deploy.prototxt',
    'data/mobilenet/mobilenet.caffemodel': 'https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet.caffemodel',
    'data/mobilenet/synset.txt': 'https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt',
    'data/haarcascades/haarcascade_frontalface_default.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml',
    'data/haarcascades/haarcascade_eye.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml'
}

for path, url in files.items():
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        try:
            urllib.request.urlretrieve(url, path)
            print("Done.")
        except Exception as e:
            print(f"Failed to download {path}: {e}")
    else:
        print(f"{path} already exists.")
