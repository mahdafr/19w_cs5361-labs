import classwork.dataset as d
from lab5 import cnn, dnn

# writes results to file
def report(test, score, f):
    f.write("\n" + test + "\n")
    f.write('\tTest loss:\t' + str(score[0]))
    f.write('\n\tTest accuracy:\t' + str(score[1]))

if __name__ == "__main__":
    # FIXME for each test IMPLEMENT AND DO THIS TEST
    mods = 'INITIALIZER=he_normal, REGULARIZER=l1_l2(0.001)'
    # run the tests
    f = open("lab5.txt", "a")
    f.write('\n===================='+mods+'====================')
    for i in [True, False]:     # to run CNN then DNN
        data = d.Dataset(using_keras=i)
        # data.print_dims()
        if not i:        # dnn for gamma/sp
            report("Gamma Ray Dataset", dnn.gamma(*data.get("gamma")), f)
            report("Solar Particle Dataset", dnn.solar(*data.get("solar")), f)
        else:        # cnn for mnist/cifar10
            report("MNIST Dataset", cnn.mnist(*data.get("mnist")), f)
            report("CIFAR-10 Dataset", cnn.cifar(*data.get("cifar10")), f)
    f.close()
