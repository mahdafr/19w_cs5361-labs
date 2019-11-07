import classwork.dataset as d

if __name__ == "__main__":
    data = d.Dataset(using_keras=True)
    f = open("lab5.txt", "a")
    f.write('\n====================RUN====================')
    # dnn for gamma/sp
    x_train, y_train, x_test, y_test = data.get("gamma")
    x_train, y_train, x_test, y_test = data.get("solar")
    # cnn for mnist/cifar10
    x_train, y_train, x_test, y_test = data.get("mnist")
    # x_train, y_train, x_test, y_test = data.get("cifar")
    f.close()
