
import dlib
import cv2
import numpy as np
from tqdm import tqdm
import os
from PIL import Image

if __name__ == "__main__":
    detector = dlib.get_frontal_face_detector()
    img_path = "img.jpg"

    img = cv2.imread(img_path)
    # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    # 偵測人臉
    face_rects = detector(img, 0)

    # 取出所有偵測的結果
    if(face_rects):
        for i, d in enumerate(face_rects):
            x1 = np.maximum(d.left(), 0)
            y1 = d.top()
            x2 = np.maximum(d.right(), 0)
            y2 = d.bottom()

        rec_frame_array = img[y1:y2, x1:x2]

    
        # rec_frame = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
    # cut_img = Image.fromarray(rec_frame_array, mode="RGB") # np.array to PIL Image
    # cut_img.save(save_path + id + ".jpg")
    # rec_frame_array = cv2.cvtColor(rec_frame_array, cv2.COLOR_RGB2BGR)
    cv2.imshow("test", rec_frame_array)
    # cv2.imwrite("rec_img.jpg", rec_frame)
    cv2.imwrite("cut_img.jpg", rec_frame_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()