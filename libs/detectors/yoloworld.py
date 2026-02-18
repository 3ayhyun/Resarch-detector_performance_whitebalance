import torch
import ultralytics
from .base_detector import BaseDetector

class YOLOWorld(BaseDetector):
    def __init__(self, config):
        self.config = config
        self.weight_path = config.get('weight_path', 'assets/weights/yolo8s-world.pt')
        self.conf = config.get('conf', 0.25)
        self.iou = config.get('iou', 0.7)
        self.classes = config.get('classes', ['white shirts']) # 비어있는 경우 white shirtss로 초기화 
        self.imgsz = config.get('imgsz', 640)

        # yoloworld는 class를 리스트 형태로 입력받는다 
        # list 형태가 아닌 경우의 error를 방지하기 위해 list형태가 아닌 경우 list로 강제한다 
        if not isinstance(self.classes, list):
            self.classes = [cls.strip() for cls in self.classes.split(',')]
        
        self.model = ultralytics.YOLOWorld(self.weight_path)

        self.model.set_classes(self.classes)
        
        print('[YOLOWorld] Initialized')
        

    def predict(self, img):
        '''
            img를 통한 yoloworld의 detection 결과를 반환한다 
        '''
        results = self.model.predict(
            source = img,
            conf = self.conf,
            iou = self.iou,
            imgsz = self.imgsz
            )
        
        for result in results:
            bbox = result.boxes # type : torch.Tensor
        
        # type : torch.Tensor이므로 형태를 list로 변환해준다 
        # 
        bboxes = bbox.xyxy.tolist() 
        conf_scores = bbox.conf.tolist()
        class_ids = bbox.cls.tolist()

        '''
            검출된 객체들의 박스좌표, score, class_id를 묶어서 하나로 만들어주는 작업이 필요
            => dectection_reuslts = [[x1, y1, x2, y2, conf_score, class_id], ...]
            이때 x1, x2, y1, y2는 정규화된 좌표여야한다 
                - bbox_xy : [[x1, y1, x2, y2], ...]
                - conf_score : [0.x, ...]
                - class_id : [각 클래스]
        '''
        h, w = img.shape[:2]
        detection_results = []
        for bbox, score, id in zip(bboxes, conf_scores, class_ids):
            x1, y1, x2, y2 = bbox
            conf_score = score
            class_id = int(id)
            class_name = self.classes[class_id]

            # 좌표계 정규화 / 좌표는 반드시 [0,1]범위
            x1 = max(0.0, min(x1 / w, 1.0))
            x2 = max(0.0, min(x2 / w, 1.0))
            y1 = max(0.0, min(y1 / h, 1.0))
            y2 = max(0.0, min(y2 / h, 1.0))

            det_result = [x1, y1, x2, y2, conf_score, class_name]
            detection_results.append(det_result)

        return detection_results
    