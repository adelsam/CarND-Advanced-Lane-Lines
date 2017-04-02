class Line(object):
    def __init__(self, name):
        self.name = name
        self.detected = False
        self.fit = None
