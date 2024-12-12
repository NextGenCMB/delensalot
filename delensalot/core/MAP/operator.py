class base:
    def __init__(self, **operator_desc):
        self.operation = operator_desc['operation']


    def act(self, obj):
        self.operation(obj)