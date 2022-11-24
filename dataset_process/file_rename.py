import random
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2

emotion = input("Emotion : ")

original_path = "LIRIS\\training_data\\valid\\" + emotion + "\\"
ids = next(os.walk(original_path))[2] # => ('LIRIS\\train\\disgust\\', [],[image.jpg ...])

for n, img_name in tqdm(enumerate(ids)):
    img_path = os.path.join(original_path, img_name)
    new_path = "LIRIS\\training_data\\valid\\"  + str(n) + ".jpg"
    os.rename(img_path, new_path)
    print(new_path)