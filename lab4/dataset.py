from _dataset import read_gamma, read_solar, read_mnist


# Holds all datasets for testing/training
class Data(object):
    def __init__(self):
        self.mnist_X = None
        self.mnist_Y = None
        self.mnist_x = None
        self.mnist_y = None
        self.gamma_X = None
        self.gamma_Y = None
        self.gamma_x = None
        self.gamma_y = None
        self.solar_X = None
        self.solar_Y = None
        self.solar_x = None
        self.solar_y = None
        self.load()

    # reads data from file(s) for program use
    def load(self):
        self.gamma_X, self.gamma_Y, self.gamma_x, self.gamma_y = read_gamma.read()
        self.mnist_X, self.mnist_Y, self.mnist_x, self.mnist_y = read_mnist.read()
        self.solar_X, self.solar_Y, self.solar_x, self.solar_y = read_solar.read()
        print('')

    # if randomizing tests, use this function
    def get(self, i):
        switcher = {
            0: self.gamma(),
            1: self.mnist(),
            2: self.solar(),
        }
        return switcher.get(i)

    # which dataset are we using?
    def using(self,i):
        switcher = {
            0: 'Gamma Ray',
            1: 'MNIST',
            2: 'Solar',
        }
        return switcher.get(i, 'rip')

    # the number of datasets in use
    def sets(self):
        return 3

    def gamma(self):
        return self.gamma_X, self.gamma_Y, self.gamma_x, self.gamma_y

    def mnist(self):
        return self.mnist_X, self.mnist_Y, self.mnist_x, self.mnist_y

    def solar(self):
        return self.solar_X, self.solar_Y, self.solar_x, self.solar_y
