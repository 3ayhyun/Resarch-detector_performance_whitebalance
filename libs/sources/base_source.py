source_registry = {}

class BaseSource:
    '''
        BaseSource를 상속받은 클래스가 정의될때(만들어질때) source_registry에 클래스를 저장하여
        build에서 클래스 이름만으로 클래스를 호출할수 있게한다 
    '''
    def __init_subclass__(cls, **kwargs): # base를 상속받는 순간 자동으로 호출된다 
        super().__init_subclass__(**kwargs)
        # {'ImageSource' : <class 'ImageSource'>, ...} (key: 클래스 이름, value: 클래스)의형태로 저장된다 
        source_registry[cls.__name__] = cls