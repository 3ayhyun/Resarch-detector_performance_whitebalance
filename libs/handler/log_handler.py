import os, csv
from .base_handler import BaseHandler

class LogHandler(BaseHandler):
    def __init__(self, config):
        self.config = config
        self.pixel = config.get('pixel', True)
        self.dir = config.get('dir', './logs')
        self.file_name = config.get('file_name', 'results_log.csv')
        self.file_path = os.path.join(self.dir, self.file_name)
        self.log_file = open(self.file_path, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.log_file)
        
        self.writer.writerow(['frame_id', 'x1', 'y1', 'x2', 'y2', 'score', 'class'])

        print('[LogHandler] Initialized', flush=True)

    def write(self, result):
        # result: {'frame_id' : frame_id,'image' : image,'detection_results' : detection_results,'org_img_shape' : org_img_shape}
        '''
            writer를 이용하여 로그를 기록해나간다 
            detections안에 해당 frame의 모든 detection결과가 들어있다 [[detection], [], ...]
            모든 detection을 한줄씩 csv로 작성해야함 
        '''
        frame_id = result.get('frame_id')
        if frame_id is None:
            self.file_close()
            return 
        
        org_img_shape = result.get('org_img_shape')
        detections = result.get('detection_results')
        
        for det in detections:
            if self.pixel:
                coord = det[:4]
                score, class_id = det[4:]
                x1, y1, x2, y2 = self.to_pixel(coord, org_img_shape)
            else:
                x1, y1, x2, y2, score, class_id = det
            log = [frame_id, x1, y1, x2, y2, score, class_id]
            self.writer.writerow(log)
        
    def file_close(self):
        self.log_file.close()
        print('[LogHandler] File Closed')

    def to_pixel(self, coord, org_img_shape):
        '''
            정규화되어있는 좌표를 원본 이미지 기준 픽셀좌표로 변환
        '''
        x1, y1, x2, y2 = coord
        org_h, org_w = org_img_shape
        coord_pixel = [
            x1 * org_w,
            y1 * org_h,
            x2 * org_w,
            y2 * org_h
        ]

        return coord_pixel
        