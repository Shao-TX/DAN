import random
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2

emotion = input("Emotion : ")

original_path = "LIRIS\\cut_image\\" + emotion + "\\"
ids = next(os.walk(original_path))[2] # => ('LIRIS\\train\\disgust\\', [],[image.jpg ...])
a = len(ids)
b = 0.2 * a
random.seed(1520)
result = random.sample(range(0, a), int(b))

# # print(ids)
# print(a)
# print(b)
# print(result)
# print(path2)

for n, img_id in tqdm(enumerate(result)):
    img_id = str(img_id)
    img_name = img_id + ".jpg"
    img_path = os.path.join(original_path, img_name)

    img = cv2.imread(img_path) # 讀取原圖位置
    cv2.imwrite("LIRIS\\cut_image\\valid\\" + emotion + "\\" + img_id + ".jpg", img) # 存放到新的位置

    os.remove(img_path)