handler_registry = {}

class BaseHandler:
    def __init_subclass__(cls, **kwargs):
        super.__init_subclass__(**kwargs)
        handler_registry[cls.__name__] = cls