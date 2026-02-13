import os, csv

class LogHandler:
    def __init__(self, config):
        self.config = config
        self.dir = config.get('dir', './logs')
        self.file_name = config.get('file_name', 'results_log.csv')
        
        self.file_path = os.path.join(self.dir, self.file_path)
        self.log_file = open(self.file_path, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.log_file)
        
        self.writer.writerow(['frame_id', 'x1', 'y1', 'x2', 'y2', 'score', 'class'])

        print('[LogHandler] Initialized')

    def log_write(self, frame_id, detections):
        '''
            writer를 이용하여 로그를 기록해나간다 
            detections안에 해당 frame의 모든 detection결과가 들어있다 [[detection], [], ...]
            모든 detection을 한줄씩 csv로 작성해야함 
        '''
        for det in detections:
            x1, y1, x2, y2, score, class_id = detections
            log = [frame_id, x1, y1, x2, y2, score, class_id]
            self.writer.writerow(log)
        
    def file_close(self):
        self.log_file.close()
        print('[LogHandler] File Closed')