from keras import regularizers
from keras.initializers import glorot_normal, he_normal, he_uniform, glorot_uniform
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, SGD, Adam\

batch_size = 32
epochs = 20

def gamma(X,Y,x,y):
    print('Gamma Ray DNN')
    model = Sequential()
    # FIXME change to use baseline model
    # model = _baseline_gamma(model, (X.shape[1],))
    model = _tester_gamma(model, (X.shape[1],))
    return fit_and_eval(X,Y,x,y,model)

def solar(X,Y,x,y):
    print('Solar Particle DNN')
    model = Sequential()
    # FIXME change to use baseline model
    model = _baseline_solar(model, (X.shape[1],))
    # model = _tester_solar(model, (X.shape[1],))
    return fit_and_eval(X,Y,x,y,model)


def fit_and_eval(X,Y,x,y,model):
    model.fit(X, Y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x, y))
    return model.evaluate(x, y, verbose=2)


""" FOR GAMMA """
def _tester_gamma(model, input_shape):
    model.add(Dense(10, activation='elu', input_shape=input_shape))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.01),
                  metrics=['accuracy'])
    return model

def _baseline_gamma(model, input_shape):
    model.add(Dense(10, activation='relu', input_shape=input_shape))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.01),
                  metrics=['accuracy'])
    return model


""" FOR SOLAR """
def _tester_solar(model, input_shape):
    model.add(Dense(30, activation='elu', input_shape=input_shape))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.summary()
    model.compile(loss='mean_squared_error',
                  optimizer=Adam(lr=0.01),
                  metrics=['mse'])
    return model

def _baseline_solar(model, input_shape):
    model.add(Dense(30, activation='relu', input_shape=input_shape))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.summary()
    model.compile(loss='mean_squared_error',
                  optimizer=Adam(lr=0.01),
                  metrics=['mse'])
    return model
