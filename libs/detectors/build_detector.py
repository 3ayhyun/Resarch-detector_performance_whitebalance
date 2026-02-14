from .base_detector import detector_registry

def build_detector(config):
    # detector_registry에 있는 class를 class_name인 key값을 통해 불러온다 
    detector_name = config.get('type', 'YOLOWorld')
    detector = detector_registry.get(detector_name) # .get을 쓰면 detector_name이 없는 경우 None을 반환하게된다 
    if detector is None:
        raise ValueError(f'can not found detector: {detector_name}')
    
    return detector(config)