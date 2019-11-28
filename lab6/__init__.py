import _dataset.dataset as dataset
import numpy as np
from keras.layers import LSTM, Conv1D, Dense, Dropout
from keras import Sequential

# default values:
# lookback is 12 hours, delay is 3 hours, and
# prediction window is 1 hour
def baseline(data, b=144, d=36, p=12):
    x = data[:,-1]
    X = []; Y = []
    for i in range(4,data.shape[1]):
        for ex in range(data.shape[0]):
            X.append(x[ex:ex + b])
            Y.append(np.mean(x[b + d:b + d + p]))
        print("Done with column=" + str(i))
        np.save('X=' + str(i) + '.npy', X)
        np.save('Y=' + str(i) + '.npy', Y)

# default values:
# lookback is 12 hours, delay is 3 hours, and
# prediction window is 1 hour
# version=0 uses LSTM
# version=1 uses CONV1D
def predict_keras(x, b=144, d=36, p=12,vers=0):
    model = None
    if vers==0:     # using LSTM
        # add dim3
        model = Sequential([LSTM(64),LSTM(32),LSTM(16),Dropout(0.25),Dense(1, activation='sigmoid')])
    else:       # using CONV1D
        model = Sequential(Conv1D())##FIXME))        model.compile()
    model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    return model.predict(x,batch_size=32,verbose=2)

# default values:
# lookback is 12 hours, delay is 3 hours, and
# prediction window is 1 hour
def predict_sklearn(x, b=144, d=36, p=12):
    return

if __name__ == "__main__":
    D = dataset.Dataset(load=False)
    xrp = D.xrp_load()   # get the good stuff
    x = xrp[:,-1]   # chop that shit
    baseline(xrp)
    # predict_keras(xrp,vers=0)
    # predict_sklearn(xrp)
