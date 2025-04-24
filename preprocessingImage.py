import cv2
import imutils
import numpy as np
import os
from keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        # Lưu ảnh đã được định dạng
        self.dataFormat = dataFormat

    def preprocess(self, image):
        # Chuyển đổi ảnh thành mảng numpy
        return img_to_array(image, data_format=self.dataFormat)  


class AspectAwarePreprocesser:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # Lấy chiều cao và chiều rộng của ảnh
        (h, w) = image.shape[:2]
        dw = 0
        dh = 0
        
        # Nếu chiều rộng nhỏ hơn chiều cao
        if w < h:
            # Giữ tỷ lệ và thay đổi kích thước chiều rộng
            image = imutils.resize(image, width=self.width, inter=self.inter)
            dh = int((image.shape[0] - self.height) / 2.0)  # Tính toán chiều cao cắt
        else:
            # Giữ tỷ lệ và thay đổi kích thước chiều cao
            image = imutils.resize(image, height=self.height, inter=self.inter)
            dw = int((image.shape[1] - self.width) / 2.0)  # Tính toán chiều rộng cắt
        
        # Cắt ảnh sao cho đúng tỷ lệ
        (h, w) = image.shape[:2]
        image = image[dh:h-dh, dw:w-dw]
        
        # Thay đổi kích thước ảnh về kích thước mong muốn
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
    
class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        # Lưu ảnh tiền xử lý
        self.preprocessors = preprocessors

        # Nếu bước tiền xử lý là None thì khởi tạo danh sách rỗng
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        # Khởi tạo danh sách các đặc trưng và nhãn
        data = []
        labels = []

        # Lặp qua tất cả ảnh đầu vào
        for (i, imagePath) in enumerate(imagePaths):
            # Nạp ảnh và trích xuất nhãn từ đường dẫn định dạng
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            # check to see if our preprocessors are not None
            if self.preprocessors is not None:
                # Lặp qua tất cả tiền xử lý và áp dụng cho mỗi ảnh
                for p in self.preprocessors:
                    image = p.preprocess(image)
            # Mỗi ảnh được xử lý là vector đặc trưng bằng cách
            # cập nhật danh sách dữ liệu cùng với nhãn
            data.append(image)
            labels.append(label)

            # Hiển thị ảnh cập nhật
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
               print("[INFO] Đã xử lý {}/{}".format(i + 1,len(imagePaths)))
                # Trả về dữ liệu kiểu tuple gồm dữ liệu và nhãn
        return (np.array(data), np.array(labels))

