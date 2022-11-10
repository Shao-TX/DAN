import dlib
import cv2

from PIL import Image
import numpy as np
import cv2

import torch
from torchvision import transforms

from networks.dan import DAN



class Model():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transforms = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                ])
        self.labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']

        self.model = DAN(num_head=4, num_class=8, pretrained=False)
        checkpoint = torch.load('./checkpoints/pre_trained_affecnet8_epoch5_acc0.6209.pth',
            map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        self.model.to(self.device)
        self.model.eval()

    def dlib_detect(self, read_img):

        img = Image.fromarray(read_img) # np.array to PIL Image

        img = self.data_transforms(img)
        img = img.view(1,3,224,224)
        img = img.to(self.device)

        with torch.set_grad_enabled(False):
            out, _, _ = self.model(img)
            _, pred = torch.max(out,1)
            index = int(pred)
            label = self.labels[index]

            return label

if __name__ == "__main__":
    detector = dlib.get_frontal_face_detector()

    cap = cv2.VideoCapture(0)

    model = Model()

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)

        # 偵測人臉
        face_rects = detector(frame, 0)

        # 取出所有偵測的結果
        if(face_rects):
            for i, d in enumerate(face_rects):
                x1 = np.maximum(d.left(), 0)
                y1 = d.top()
                x2 = np.maximum(d.right(), 0)
                y2 = d.bottom()

            # 將 frame 中的人臉標示方框
            rec_frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
            
            # 將 frame 中的人臉標示方框的圖像取出
            rec_frame_array = rec_frame[y1:y2, x1:x2]

            label = model.dlib_detect(rec_frame_array)

            cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
            # print(f'emotion label: {label}')
            # cv2.imshow("Rec", rec_frame_array)

        # 顯示結果
        cv2.imshow("Face Detection", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()