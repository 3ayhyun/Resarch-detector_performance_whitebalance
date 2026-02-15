import multiprocessing as mp
import queue
import cv2
from libs.handler.build_handler import build_handler
'''
    log기록과 시각화를 실행하는 파이프라인 
'''
class OutputPipeline(mp.Process):
    def __init__(self, config, results_queue):
        super().__init__()
        self.config = config
        self.results_queue = results_queue
        # config의 handler부분에는 log, screen 두개이므로 두 핸들러를 모두 한 리스트에 저장해서 run에서 불러오도록 한다 
        self.handlers = []
        '''for h_config in config['handler']:
            handler = build_handler(h_config)
            self.handlers.append(handler)'''

    def run(self):
        for h_config in self.config['handler']:
            handler = build_handler(h_config)
            self.handlers.append(handler)

        while True:

            try:
                result = self.results_queue.get(timeout=1) # 호출될때마다 다음걸 하나 꺼내온다 (while로 반복하여 꺼내고 다음거꺼내고 ...)
            except queue.Empty: # result에 다음 값이 업데이트되어있지 않으면 empty 예외가 발생한다 
                continue

            # 저장형태 : [{'frame_id' : frame_id, 'detection_results' : detection_results}, ... ]
            frame_id = result.get('frame_id')
            image = result.get('image')
            detection_results = result.get('detection_results')

            if image is None:
                break

            # screen handler, log handler를 이용하여 화면출력, 로그 기록 
            for handler in self.handlers:
                if hasattr(handler, 'draw'): # handler가 draw라는 함수를 가지고 있으면 screen handler
                    handler.draw(image, detection_results)
                if hasattr(handler, 'write'):
                    handler.write(frame_id, detection_results)
            
            if cv2.waitKey(1) &0xFF == ord('q'):
                break