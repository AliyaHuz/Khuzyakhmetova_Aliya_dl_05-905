
class Normalize(object):
    def __init__(self, mean=128, var=255):
        """
        :param mean (int or tuple): среднее значение (пикселя), которое необходимо вычесть.
        :param var (int): значение, на которое необходимо поделить.
        """
        pass

    def __call__(self, images):
        return (images-self.mean)/self.var

class View(object):
    def __init__(self):
        pass
    def __call__(self, images):
        return images.reshape(images.shape[0], -1)