# import cv2
# from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
# import torch
# from torchvision import transforms
# import numpy as np
# from PIL import Image
# import time
# from firebase import db
# frame_size = (640,480)
# IMG_PATH = './data/test_images'
# DATA_PATH = './data'

# def trans(img):
#     transform = transforms.Compose([
#             transforms.ToTensor(),
#             fixed_image_standardization
#         ])
#     return transform(img)

# def load_faceslist():
#     if device == 'cpu':
#         embeds = torch.load(DATA_PATH+'/faceslistCPU.pth')
#     else:
#         embeds = torch.load(DATA_PATH+'/faceslist.pth')
#     names = np.load(DATA_PATH+'/usernames.npy')
#     return embeds, names

# def inference(model, face, local_embeds, threshold = 1):
#     #local: [n,512] voi n la so nguoi trong faceslist
#     embeds = []
#     # print(trans(face).unsqueeze(0).shape)
#     embeds.append(model(trans(face).to(device).unsqueeze(0)))
#     detect_embeds = torch.cat(embeds) #[1,512]
#     # print(detect_embeds.shape)
#                     #[1,512,1]                                      [1,512,n]
#     norm_diff = detect_embeds.unsqueeze(-1) - torch.transpose(local_embeds, 0, 1).unsqueeze(0)
#     # print(norm_diff)
#     norm_score = torch.sum(torch.pow(norm_diff, 2), dim=1) #(1,n), moi cot la tong khoang cach euclide so vs embed moi
    
#     min_dist, embed_idx = torch.min(norm_score, dim = 1)
#     print(min_dist*power, names[embed_idx])
#     # print(min_dist.shape)
#     if min_dist*power > threshold:
#         return -1, -1
#     else:
#         return embed_idx, min_dist.double()

# def extract_face(box, img, margin=20):
#     face_size = 160
#     img_size = frame_size
#     margin = [
#         margin * (box[2] - box[0]) / (face_size - margin),
#         margin * (box[3] - box[1]) / (face_size - margin),
#     ] #tạo margin bao quanh box cũ
#     box = [
#         int(max(box[0] - margin[0] / 2, 0)),
#         int(max(box[1] - margin[1] / 2, 0)),
#         int(min(box[2] + margin[0] / 2, img_size[0])),
#         int(min(box[3] + margin[1] / 2, img_size[1])),
#     ]
#     img = img[box[1]:box[3], box[0]:box[2]]
#     face = cv2.resize(img,(face_size, face_size), interpolation=cv2.INTER_AREA)
#     face = Image.fromarray(face)
#     return face

# if __name__ == "__main__":
#     prev_frame_time = 0
#     new_frame_time = 0
#     power = pow(10, 6)
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     print(device)

#     model = InceptionResnetV1(
#         classify=False,
#         pretrained="casia-webface"
#     ).to(device)
#     model.eval()

#     mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)

#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
#     embeddings, names = load_faceslist()
    
#     # start_time = None
#     # recognized_name = None
#     # saved_image = None

#     while cap.isOpened():
#         isSuccess, frame = cap.read()
#         if isSuccess:
#             boxes, _ = mtcnn.detect(frame)
#             if boxes is not None:
#                 for box in boxes:
#                     bbox = list(map(int,box.tolist()))
#                     face = extract_face(bbox, frame)
#                     idx, score = inference(model, face, embeddings)
#                     if idx != -1:
#                         frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
#                         score = torch.Tensor.cpu(score[0]).detach().numpy()*power
#                         frame = cv2.putText(frame, names[idx] + '_{:.2f}'.format(score), (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)
                        
#                         # if recognized_name == name:
#                         #       if time.time() - start_time >= 5:
#                         #         # Sau 5 giây, chụp ảnh
#                         #         saved_image = frame.copy()
#                         #         cv2.imwrite(f"{IMG_PATH}/{name}_captured.jpg", saved_image)
#                         #         print(f"📸 Đã chụp ảnh: {name}_captured.jpg")
#                         # else:
#                         #     recognized_name = name
#                         #     start_time = time.time()
                            
                   
#                     # #     #đẩy fire basebase                    
#                     #     user_ref = db.reference("recognized_users")
#                     #     user_ref.set({
#                     #         "name": names[idx],
#                     #         "timestamp": time.time()
#                     #         })
#                     else:
#                         frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
#                         frame = cv2.putText(frame,'Unknown', (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)
#                         # recognized_name = None
#                         # start_time = None
#             new_frame_time = time.time()
#             fps = 1/(new_frame_time-prev_frame_time)
#             prev_frame_time = new_frame_time
#             fps = str(int(fps))
#             cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

#         cv2.imshow('Face Recognition', frame)
#         if cv2.waitKey(1)&0xFF == 27:
#             break

#     cap.release()
#     cv2.destroyAllWindows()
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelBinarizer
from imutils import paths
import os

# Hàm để hiển thị ảnh và nhãn dự đoán
def display_image_with_prediction(image, true_label, predicted_label):
    plt.imshow(image)
    plt.title(f"True: {true_label}, Pred: {predicted_label}")
    plt.axis('off')
    plt.show()

# Tải mô hình đã huấn luyện
model = load_model("fine_tuned_facenet_model.h5")

# [1] Đọc và tiền xử lý ảnh kiểm tra
imagePaths = list(paths.list_images("data"))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

# Preprocessing ảnh
from preprocessingImage import ImageToArrayPreprocessor, AspectAwarePreprocesser, SimpleDatasetLoader
aap = AspectAwarePreprocesser(160, 160)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=200)
data = data.astype("float32") / 255.0

# One-hot encoding cho nhãn
LabelBinarizer_obj = LabelBinarizer()
labels = LabelBinarizer_obj.fit_transform(labels)

# [2] Dự đoán cho một ảnh kiểm tra
for i in range(len(data)):
    # Dự đoán nhãn cho ảnh kiểm tra
    pred = model.predict(np.expand_dims(data[i], axis=0))  # Dự đoán 1 ảnh
    
    # Lấy nhãn dự đoán
    predicted_class = classNames[np.argmax(pred)]  # Nhãn dự đoán
    true_class = classNames[np.argmax(labels[i])]  # Nhãn thực tế
    
    # Hiển thị ảnh và so sánh với nhãn thực tế
    display_image_with_prediction(data[i], true_class, predicted_class)
