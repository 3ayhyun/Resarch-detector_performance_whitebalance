from .base_handler import handler_registry

# log handler와 screen handler 모두 반환 
def build_handler(config):
    class_name = config.get('type')
    handler = handler_registry[class_name]
    return handler(config)