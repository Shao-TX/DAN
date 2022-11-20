import cv2
import os
import io
from google.cloud import vision_v1
from google.cloud.vision_v1 import types
from tqdm import tqdm

#%%
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'angelic-bee-323206-91fcd6b54e96.json'

#%%
client = vision_v1.ImageAnnotatorClient()

#%%

if __name__ == "__main__":

    emotion = input("Emotion : ")


    folder_path = "LIRIS_Image\\Original\\" + emotion
    ids = next(os.walk(folder_path + "\\"))[2] # => ('LIRIS\\train\\disgust\\', [],[image.jpg ...])
    save_path = "LIRIS_Image\\Filter\\" + emotion + "\\"
    normal_path = "LIRIS_Image\\Filter\\" + emotion + "_normal\\"

    for id, img_path in tqdm(enumerate(ids)):
        id = str(id)
        img_path = os.path.join(folder_path, img_path)
        read_img = cv2.imread(img_path)

        img = cv2.imencode('.jpg', read_img)[1].tobytes()

        image = vision_v1.Image(content=img)

        likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY')

        response_face = client.face_detection(image=image)

        for face_detection in response_face.face_annotations:
            joy_mood = likelihood_name[face_detection.joy_likelihood]
            sorrow_mood = likelihood_name[face_detection.sorrow_likelihood]
            surprise_mood = likelihood_name[face_detection.surprise_likelihood]
            anger_mood = likelihood_name[face_detection.anger_likelihood]


        if(joy_mood == 'LIKELY' or joy_mood == 'VERY_LIKELY' or joy_mood == 'POSSIBLE'):
            cv2.imwrite("LIRIS_Image\\Filter\\happy\\" + id + ".jpg", read_img)
            print("Save to Happy : ", joy_mood)
            print("ID : ", id)
        elif(sorrow_mood == 'LIKELY' or sorrow_mood == 'VERY_LIKELY' or sorrow_mood == 'POSSIBLE'):
            cv2.imwrite("LIRIS_Image\\Filter\\sad\\" + id + ".jpg", read_img)
            print("Save to Sad : ", sorrow_mood)
            print("ID : ", id)
        elif(surprise_mood == 'LIKELY' or surprise_mood == 'VERY_LIKELY' or surprise_mood == "POSSIBLE"):
            cv2.imwrite("LIRIS_Image\\Filter\\surprise\\" + id + ".jpg", read_img)
            print("Save to Surprise : ", surprise_mood)
            print("ID : ", id)
        else :
            cv2.imwrite(normal_path + emotion + "_" +id + ".jpg", read_img)
            print("Save to Normal : ", sorrow_mood)
            print("ID : ", id)