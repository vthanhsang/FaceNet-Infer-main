import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras_facenet import FaceNet
from imutils import paths
from tensorflow.keras.models import save_model
import os

# Preprocessing ảnh
from preprocessingImage import ImageToArrayPreprocessor, AspectAwarePreprocesser, SimpleDatasetLoader

# [1] Load ảnh
print("[INFO] Đang nạp ảnh...")
imagePaths = list(paths.list_images("data"))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

# [2] Tiền xử lý ảnh về 160x160
aap = AspectAwarePreprocesser(160, 160)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=200)
data = data.astype("float32") / 255.0

# [3] Tách tập train/test
Data_train, Data_test, Label_train, Label_test = train_test_split(data, labels, test_size=0.20, random_state=42)

# [4] One-hot encoding cho nhãn
Label_train = LabelBinarizer().fit_transform(Label_train)
Label_test = LabelBinarizer().fit_transform(Label_test)

# [5] Tải mô hình FaceNet từ keras_facenet
print("[INFO] Đang tải mô hình FaceNet pretrained...")
# Tải mô hình FaceNet đã được huấn luyện trước
embedder = FaceNet()

# Đóng băng các layer của mô hình FaceNet
for layer in embedder.model.layers:
    layer.trainable = False

# Thêm các lớp phân loại mới
x = embedder.model.output
x = Dense(512, activation='relu')(x)
x = Dense(len(classNames), activation='softmax')(x)
model = Model(inputs=embedder.model.input, outputs=x)

# Biên dịch và huấn luyện mô hình
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(Data_train, Label_train, epochs=10, batch_size=32, validation_data=(Data_test, Label_test))

# Lưu mô hình đã tinh chỉnh
save_model(model, 'fine_tuned_facenet_model.h5')

# ten 2.19.0 