import os
import glob
import cv2
import time

class ImageSource:
    def __init__(self, config):
        self.config = config
        self.assets_path = config.get('assets_path', 'assets/images')
        self.pattern = config.get('pattern', '*.jpg')

        self.files = glob.glob(os.path.join(self.assets_path, self.pattern)) # glob.glob를 통해 pattern 형태의 파일명을 리스트로 반환한다 
        # 파일을 숫자순으로 정렬
        self.files.sort()

        self.idx = 0

        self.fps = config.get('fps', 30)
        self.interval = 1.0 / self.fps # 1초에 fps장의 이미지를 보여주기 위한 이미지 한장을 보여주는 시간
        self.last_time = time.perf_counter()


    def read(self):
        '''
            read 호출시 해당 프레임의 이미지를 읽어야한다 
            => _init__에서 idx=0으로 초기화, 마지막에 +=1
                idx는 이미지 순서이자 frame_id를 의미

            return frame_id, image
        '''
        # 1/fps 만큼의 시간이 지나지 않았다면 남은 시간만큼 sleep을 건다 
        now_time = time.perf_counter()
        time_gap = now_time - self.last_time
        if time_gap < self.interval:
            time.sleep(self.interval - time_gap)
        self.last_time = time.perf_counter()

        if self.idx < len(self.files):
            image = cv2.imread(self.files[self.idx]) # cv2를 이용했으므로 BGR 형태
            frame_id = self.idx
            self.idx += 1 # 다음 read 호출시 다음프레임 이미지를 반환하기 위해 
        else: # 더이상 읽어들일 파일이 없는 경우
            frame_id, imagea = None, None

        return frame_id, image