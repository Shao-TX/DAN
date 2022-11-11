import cv2
import os

video_classes = ["disgust", "fear", "happy", "sad", "surprise"]

def file_count(dir):
    file_list = []
    for file_name in os.listdir(dir):
        file_list.append(file_name)

    return file_list

def cut(file_name, emotion):
    file_path = os.path.join("dataset", "video", emotion, file_name)
    cap = cv2.VideoCapture(file_path)
    i = 1
    while(True):
        ret, frame = cap.read()

        if cv2.waitKey(1) == ord('q') or ret == False : break

        cv2.imshow('video', frame)

        save_name = file_name.split('.')
        save_name =  save_name[0] + "_" + str(i)  + ".jpg"
        save_path = os.path.join("dataset", "image", emotion, save_name)

        cv2.imwrite(save_path, frame)
        i = i + 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    emotion = input("Emotion : ")
    dir = "dataset/video/" + emotion
    file_list = file_count(dir)

    for file_name in file_list:
        print(file_name)
        cut(file_name, emotion)