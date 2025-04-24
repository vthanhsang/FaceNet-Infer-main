import numpy as np
from keras_facenet import FaceNet
from imutils import paths
import cv2
import os

embedder = FaceNet()

print("[INFO] Đang tạo danh sách embedding khuôn mặt...")
imagePaths = list(paths.list_images("data/test_images"))
knownEmbeddings = []
knownNames = []

for imagePath in imagePaths:
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (160, 160))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    # Lấy embedding
    embedding = embedder.embeddings(image)
    knownEmbeddings.append(embedding[0])
    knownNames.append(name)

# Lưu xuống file
np.save("data/faceslist.npy", knownEmbeddings)
np.save("data/usernames.npy", knownNames)
print("[INFO] Đã lưu xong embeddings và tên người dùng!")
