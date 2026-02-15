import os, sys
import yaml
import argparse
import multiprocessing as mp
# groundingdino의 경롤ㄹ python path에 추가하여 사용할 수 있게 해야한다 
dino_path = os.path.join(os.path.dirname(__file__), 'libs/detectors/GroundingDINO')
sys.path.append(dino_path)
from libs.detectors import build_detector
from libs.sources import build_source
from predict.predict_pipeline import PredictPipeline
from output.output_pipeline import OutputPipeline

if __name__ == '__main__':
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    results_queue = mp.Queue() # detection_resutls를 저장할 공간  => predict의 결과를 저장하고 이를 output_pipeline에서 사용

    parser = argparse.ArgumentParser() # 명령줄을 읽는다 
    parser.add_argument('-c', '--config', type = str, default = 'configs/yoloworld.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    predict = PredictPipeline(config, results_queue)
    output = OutputPipeline(config['output'], results_queue)

    predict.start()
    output.start()
    # predict파이프라인과 output파이프라인 프로세스가 끝날때까지 main은 종료하지 않고 기다린다 
    # 멀티프로세스이므로 따로 프로세스가 실행되어 main이 먼저끝나는 경우가 생긴다
    predict.join()
    output.join()