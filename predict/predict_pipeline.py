import multiprocessing as mp
from libs.detectors.build_detector import build_detector
from libs.sources.build_source import build_source
from libs.sources.base_source import source_registry
'''
    predict를 실행하는 파이프라인 
'''
class PredictPipeline(mp.Process):
    def __init__(self, config, results_queue): 
        super().__init__()
        self.config = config
        self.results_queue = results_queue

    def run(self):
        self.detector = build_detector(self.config['predict']['detector'])
        self.source = build_source(self.config['source'])
        # 이미지가 계속 공급되는한 루프를 돌며 추론, 화면출력, 로그저장을 수행한다 
        while True:
            # source의 read()를 이용해 frame id, image를 불러옴
            frame_id, image = self.source.read()
            # 더이상 읽을 이미지가 없으면 frame id와 image는 None값을 부여받도록 되어있으므로 
            if image is None:
                self.results_queue.put(
                    {
                        'frame_id' : None,
                        'image' : None,
                        'detection_results' : None
                    }
                )
                break
            # 원본 이미지의 w, h
            org_img_shape = image.shape[:2]
            # detector의 predict()에 이미지를 입력해 추론
            detection_results = self.detector.predict(image)
            # 결과 저장 
            # log_handler에서 frame_id와 detection_results 둘다 필요하므로 둘다 저장해야함
            self.results_queue.put(
                {
                    'frame_id' : frame_id,
                    'image' : image,
                    'detection_results' : detection_results,
                    'org_img_shape' : org_img_shape
                }
            )
            

            