from libs.sources.base_source import source_registry

'''
    source파일이 추가되면 사용해야하는 source파일만 사용하게한다 
    즉 return값이 사용할 source class

    yaml파일의 type을 통해 받은 클래스를 사용하도록 return 한다 
'''
def build_source(config):
    class_name = config['type'] # class 이름과 동일 / ex) "ImageSource"

    # source_type의 이름을 가진 class를 생성
    # key가 class_name인 value(real class)를 불러온다 
    source = source_registry.get(class_name) # .get을 시용하면 class_name이 없으면 None을 반환
    if source is None: # 없는 source class인 경우 에러메시지 발생
        raise ValueError(f'can not found source name: {class_name}')

    return source(config)