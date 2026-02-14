detector_registry = {}

class BaseDetector:
    
    def __init_subclass__(cls, **kwargs):  
        super().__init_subclass__(**kwargs)
        detector_registry[cls.__name__] = cls
