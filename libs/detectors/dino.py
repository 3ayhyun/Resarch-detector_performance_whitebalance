import torch, cv2
import numpy as np
from .GroundingDINO.groundingdino.util.inference import load_model, load_image, predict
from .base_detector import BaseDetector

class GroundingDINO(BaseDetector):
    def __init__(self, config):
        self.config = config
        self.model_config_path = config.get(
            'model_config_path', 
            'libs/detectors/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
            )
        self.weight_path = config.get('weight_path', 'assets/weights/groundingdino_swint_ogc.pth')
        self.classes = config.get('classes', ['person']) 
        # grounding dino는 클래스를 문자열로 받는다 여러개일 경우 .으로 구분 ex) 'person. dogs. ...'
        # 다른 형태로 들어올 경우 'person'으로 강제후 입력형태 출력하도록 한다 
        if not isinstance(self.classes, list):
            raise TypeError ("classes must be a  list | example : ['person', 'dog', ...]")
        
        self.classes = " . ".join(self.classes)

        self.box_threshold = config.get('box_threshold', 0.3)
        self.text_threshold = config.get('text_threshold', 0.25)

        self.device = config.get('device', 'cpu')
        if self.device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else: 
            self.device = 'cpu'

        self.with_nms = config.get('with_nms', True)
        self.nms_thresh = config.get('nms_thresh', 0.5)

        # model 생성
        self.model = load_model(
            model_config_path = self.model_config_path, 
            model_checkpoint_path = self.weight_path,
            device = self.device
            )
        
        print('[GroundingDINO] Initialized')

    def predict(self, img):
        '''
            grounding dino는 이미지를 torch.Tensor형태로 받는다 
            => img를 predict하기전 torch.Tensor형태로 변환해야한다 
        '''
        # ImageNet RGB 정규화 상수 / dino는 ImageNet으로 학습하므로 정규화시 필요
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        # dino의 predict는 이미지를 RGB 형태로 받아야한다 image_source.py를 통해 받은 img는 BGR형태이므로 변환한다 
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # resize : 짧은 변이 800일때 scale구하고 긴변에 적용시 1333이 넘지않으면 짧은변이 800이 되는 scale 적용
        #          반대로 1333이 넘어가면 긴변이 1333이 되도록 scale을 바꾼다 
        h, w = image.shape[:2]
        scale = 800 / min(h,w)
        if max(h,w) * scale > 1333: # 긴 변이 1333을 넘어가면 
            scale = 1333 / max(h,w)

        scaled_h = int(h * scale)
        scaled_w = int(w * scale)

        image = cv2.resize(image, (scaled_w, scaled_h))

        # numpy 형태로 변환후 정규화 작업
        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std
        # torch.Tensor 형태인 (channel, height, weight)로
        image = np.transpose(image, (2, 0, 1))
        image = torch.Tensor(image).float()

        bboxes, scores, labels = predict(self.model, 
                                         image,
                                         self.classes, 
                                         box_threshold=self.box_threshold, 
                                         text_threshold=self.text_threshold,
                                         device=self.device)
        '''
            bboxes : bounding box( 정규화된 [cx, cy, width, height] )                                                           
            scores : 입력한 텍스트와 박스가 얼마나 유사한가 
            labels : 검출된 객체의 텍스트 리스트의 인덱스값 (["person", "car", ...])에서 bboxes가 어떤걸 의미하는지 
            
            정규화상태인 cx, cy, wdith, height를 정규화된 x1, y1, x2, y2로 변환
        '''
        detection_results = []
        for box, score, label in zip(bboxes, scores, labels):
            cx, cy, w, h = box
            # 모서리 좌표계를 구성하는값  x1, y1, x2, y2로 변환
            # 값은 반드시 [0,1] 구간이어야한다 넘어가면 안됨
            x1 = float(max(0.0, min(cx - (w / 2), 1.0)))
            y1 = float(max(0.0, min(cy - (h / 2), 1.0)))
            x2 = float(max(0.0, min(cx + (w / 2), 1.0)))
            y2 = float(max(0.0, min(cy + (h / 2), 1.0)))
            # detection_results에 추가 
            det_result = [x1, y1, x2, y2, score, label]
            detection_results.append(det_result)

        return detection_results
            