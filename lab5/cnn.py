import keras
from keras.initializers import glorot_normal, he_normal, he_uniform, glorot_uniform
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K, regularizers
from keras.optimizers import Adam, RMSprop, SGD

batch_size = 32
epochs = 20
num_classes = 10

def mnist(X,Y,x,y):
    if K.image_data_format() == 'channels_first':
        input_shape = (1, 28, 28)
    else:
        input_shape = (28, 28, 1)
    print('MNIST CNN')
    model = Sequential()
    # FIXME change to use baseline model
    # model = _baseline_mnist(model, input_shape)
    model = _tester_mnist(model, input_shape)
    return fit_and_eval(X,Y,x,y,model)

def cifar(X,Y,x,y):
    print('CIFAR-10 CNN')
    model = Sequential()
    # FIXME change to use baseline model
    # model = _baseline_cifar(model, X.shape[1:])
    model = _tester_cifar(model, X.shape[1:])
    X = X.astype('float32')
    x = x.astype('float32')
    X /= 255
    x /= 255
    return fit_and_eval(X,Y,x,y,model)


def fit_and_eval(X,Y,x,y,model,shuffle=False):
    if not shuffle:
        model.fit(X, Y,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=2,
                  validation_data=(x, y))
    else:
        model.fit(X, Y,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=2,
                  validation_data=(x, y),
                  shuffle=shuffle)
    return model.evaluate(x, y, verbose=2)


""" FOR CIFAR """
def _tester_cifar(model, input_shape):
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('elu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('elu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

def _baseline_cifar(model, input_shape):
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


""" FOR MNIST """
def _tester_mnist(model, input_shape):
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='elu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    return model

def _baseline_mnist(model, input_shape):
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    return model
