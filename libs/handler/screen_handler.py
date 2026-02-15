import numpy as np
import cv2
from .base_handler import BaseHandler

class ScreenHandler(BaseHandler):
    def __init__(self, config):
        self.config = config
        self.window_name = 'detection valify'
        # 츨력 이미지 해상도
        self.view_w = config.get('view_w', 1280)
        self.view_h = config.get('view_h', 720)

        print('[ScreenHandler] Initialized', flush=True)

    ''' image출력, 검출된객체 박스 그리기 '''    
    def draw(self, image, detections):
        '''
            프레임미다 img와 dets(detection결과)를 받아 화면에 출력, 박스 그리기
        '''
        h, w = image.shape[:2]

        if w != self.view_w or h != self.view_h:
            frame = cv2.resize(image, (self.view_w, self.view_h))
        else:
            frame = image.copy()

        ''' detections에 값이 있다면 detection box 그리기 '''
        if detections is not None:
            for det in detections:
                x1, y1, x2, y2 = self.inverse_transform(det[:4]) # 정규화된 좌표계이다 -> 출력되는 이미지기준 픽셀 좌표계로 되돌려야함 
                class_label = det[-1]
                color = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{class_label}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow(self.window_name, frame)

    ''' 종료 '''
    def close_screen(self):
        cv2.destroyAllWindows(self.window_name)

    ''' 정규화된 좌표 역변환 '''
    def inverse_transform(self, box_norm):
        x1, y1 ,x2, y2 = box_norm
        # 출력되는 이미지 크기에 맞는 박스를 만들어야하므로 출력 이미지 해상도를 기준으로 픽셀좌표계로 되돌려야한다 
        x1 *= int(self.view_w)
        y1 *= int(self.view_h)
        x2 *= int(self.view_w)
        y2 *= int(self.view_h)
        
        # 화면 밖으로 벗어나는 좌표계를 화면안으로 강제한다 
        x1 = int(max(0, min(x1, self.view_w)))
        y1 = int(max(0, min(y1, self.view_h)))
        x2 = int(max(0, min(x2, self.view_w)))
        y2 = int(max(0, min(y2, self.view_h)))

        return x1, y1, x2, y2
