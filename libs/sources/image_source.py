import os
import glob
import cv2

class ImageSource:
    def __init__(self, config):
        self.config = config
        self.assets_path = config.get('assets_path', 'asstes/images')
        self.pattern = config.get('fattern', '*jpg')

        self.files = glob.glob(os.path.join(self.assets_path, self.pattern)) # glob.glob를 통해 pattern 형태의 파일명을 리스트로 반환한다 
        # 파일을 숫자순으로 정렬
        self.files.sort()

        self.idx = 0

        self.fps = config.get('fps', 30)


    def read(self):
        '''
            read 호출시 해당 프레임의 이미지를 읽어야한다 
            => _init__에서 idx=0으로 초기화, 마지막에 +=1
                idx는 이미지 순서이자 frame_id를 의미

            return frame_id, image
        '''
        image = cv2.imread(self.files[self.idx]) # cv2를 이용했으므로 BGR 형태
        frame_id = self.idx

        self.idx += 1 # 다음번 read 호출시 다음프레임 이미지를 반환하기 위해 

        return frame_id, image