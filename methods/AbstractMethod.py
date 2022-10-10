class AbstractMethod():

    def getEncoder(self):
        raise NotImplementedError('subclasses must override getEncoder()!')

    def getHead(self):
        raise NotImplementedError('subclasses must override getHead()!')
