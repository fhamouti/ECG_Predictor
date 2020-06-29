import numpy as np
import math
import glob, os
import wfdb
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import scipy.stats as st

# Importing convolutional layers
import keras
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import LSTM,Bidirectional
from keras.optimizers import Adam
from keras.callbacks import CSVLogger

def get_data(filename):
    # P -> 1; Q -> 2; R -> 3; S -> 4; T -> 5
    record_name = os.path.basename(filename).split(".dat")[0]
    record_path = os.path.dirname(filename)

    cwd = os.getcwd()
    os.chdir(record_path)
    p_signal, _ = wfdb.rdsamp(record_name) # Read signal
    x = p_signal[:,0]

    annotation = wfdb.rdann(record_name, extension='pu0')  # Read annotation
    # List of the annotations
    annotations = list(zip(annotation.sample, annotation.symbol))

    y = np.zeros((5, len(x)), dtype=np.int)

    for i in range(len(annotations)):
        try:
            if annotations[i][1] == "p":
                y[0][annotations[i][0]] = 1
            elif (annotations[i][1] == "(" and annotations[i+1][1] == "N"):
                y[1][annotations[i][0]] = 1
            elif annotations[i][1] == "N":
                y[2][annotations[i][0]] = 1
                y[0][annotations[i][0]] = 0
            elif (annotations[i][1] == ")" and annotations[i-1][1] == "N"):
                y[3][annotations[i][0]] = 1
            elif annotations[i][1] == "t":
                y[4][annotations[i][0]] = 1

        except IndexError:
            pass
    os.chdir(cwd)
    return x , (np.transpose(y))

def normalize_signal(x):
	for i in range(x.shape[0]):
		x[i]=st.zscore(x[i], axis=0, ddof=0)
	return x

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def windowing_data(x, y,step):
    size_data = len(x)
    windowing_size = 125
    final_size_data = math.ceil(size_data/step) * step
    difference = int(final_size_data - size_data)
    # Pad zeros at the end
    x = np.pad(x, (0, difference), 'constant', constant_values=0)

    # Pad zeros to windowing
    x = np.pad(x, (windowing_size, windowing_size), 'constant',
               constant_values=0)
    y_out = np.pad(y, ((0, difference),(0,0)), 'constant', constant_values=0)
    out = np.zeros((final_size_data,step))
    ii = 0
    for i in range(windowing_size,final_size_data+windowing_size):
        out[ii,:] = x[i-windowing_size:i+windowing_size]
        ii += 1
    return out, y_out

def get_model():
    model = Sequential()
    model.add(Conv1D(filters = 16, kernel_size = 5,strides= 3, padding='same',
                     activation='relu', input_shape=(250, 1)))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters = 32, kernel_size = 5, strides= 3,
                     padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(5, activation='softmax'))
    adam = Adam(lr=1e-3, epsilon = 1e-8, beta_1 = .9, beta_2 = .999)
    model.compile(loss='mean_squared_error', optimizer=adam,
                  metrics=['accuracy'])
    print(model.summary())
    return model

def loadFiles(filenames):
    split_size = 250
    for datfile in filenames:
        annotation_file = os.path.splitext(datfile)[0] + '.pu0'
        if os.path.isfile(annotation_file):
            x, y = get_data(datfile)
            x = medfilt(x, kernel_size=7)
            x_final, y = windowing_data(x, y ,split_size)
            x_final = normalize_signal(x_final)
            try:
                xx=np.vstack(  (xx,x_final) )
                yy=np.vstack(  (yy,y) )
            except NameError:
                xx = x_final
                yy = y
    return xx, yy

def unify_data(x, number_cycles):
    n = 250 # Each cycle has 250 samples
    total_samples = number_cycles*n
    xx = np.zeros((total_samples, 1))
    i = 0
    for j in range(125,total_samples,250):
        xx[i:i+n, 0] = x[j,:,0]
        i += n
    return xx

def plot_ecg_data_trained(y_real, y_predicted, data, number_cycles):
    # Get n cyvles to plot from train and test
    n = 250
    total_samples = n * number_cycles
    vertical_lines = range(n,total_samples+n,n)
    xx = unify_data(data, number_cycles)

    colours = ['green', 'yellow', 'red', 'orange', 'cyan']
    labels_real = ['P real', 'Q real', 'R real', 'S real', 'T real']

    labels_predicted = ['P predicted', 'Q predicted',
                        'R predicted', 'S predicted', 'T predicted']

    plt.plot(xx,label='Signal', linewidth=0.7)

    # Get peak of predicted and real labels
    for i in range(len(labels_real)):
        plt.plot(y_real[:total_samples,i], label=labels_real[i],
        color = colours[i])

    for j in range(len(labels_predicted)):
        plt.plot(y_predicted[:total_samples,j],'--', label=labels_predicted[j],
        color = colours[j])
    for vertical_line in vertical_lines:
        plt.axvline(x=vertical_line, c='b', linewidth=3)

    plt.legend(loc='upper right', ncol=2)
    plt.ylabel('mV')
    plt.xlabel('Samples')
    plt.show()

def division_exception(x,y):
    result = np.zeros(5)
    for i in range(len(y)):
        if y[i] == 1 and x[i] == 0:
            result[i] = -1
        elif y[i] == 1 and x[i] == 1:
            result[i] = 1
        else: result[i] = 0
    return result

def evalution(y_actual, y_hat):
    TP = np.zeros(5)
    FP = np.zeros(5)
    TN = np.zeros(5)
    FN = np.zeros(5)
    accuracy = np.zeros(5)
    for i in range(y_hat.shape[0]):
        for j in range(y_hat.shape[1]):
            if y_actual[i][j]==y_hat[i][j]==1:
                TP[j] += 1
            if y_hat[i][j]==1 and y_actual[i][j]!=y_hat[i][j]:
                FP[j] += 1
            if y_actual[i][j]==y_hat[i][j]==0:
                TN[j] += 1
            if y_hat[i][j]==0 and y_actual[i][j]!=y_hat[i][j]:
                FN[j] += 1

    accuracy = (TP + TN)/ (TP + FP + TN + FN) * 100
    return accuracy

def expected_division(n, d):
    return n / d if d else 0

def evaluation_range_peak(y_actual, y_hat):
    labels = 5
    count = np.zeros(labels)
    real_label = np.zeros(labels)
    total_peaks = 0
    percentile = np.zeros((y_actual.shape[0],labels))

    for i in range(y_hat.shape[0]):
        for j in range(y_hat.shape[1]):
            if y_actual[i][j] == 1:
                real_label[j] += 1
                total_peaks += 1
                for l in np.arange(-10,11):
                    if 0 < i+l < y_hat.shape[0]:
                        if y_hat[i+l][j] == 1:
                            count[j] += 1
                            break
        percentile[i,:] = division_exception(count, real_label)
        count = np.zeros(labels)
        real_label = np.zeros(labels)

    mean_peaks = np.zeros((labels))
    count_peaks = 0
    discard_peaks = 0

    for m in range(percentile.shape[1]):
        for n in range(percentile.shape[0]):
            if percentile[n][m] == 1:
                count_peaks += 1
            if percentile[n][m] == -1:
                discard_peaks += 1
        mean_peaks[m] = expected_division(count_peaks,
                                          (discard_peaks+count_peaks))
        count_peaks = 0
        discard_peaks = 0
    return mean_peaks * 100


#---------------------------- MAIN ---------------------------

path_db = "qtdb/"
percent_train = 0.8
percent_test = 0.2
datfiles = glob.glob(path_db + "*.dat")

# Load data
if not os.path.isfile('x.npy'):
    x, y = loadFiles(datfiles)
    np.save('x.npy', x)
    np.save('y.npy', y)
else:
    x = np.load('x.npy')
    y = np.load('y.npy')

# Split data into train and test
size_x = int(math.ceil(percent_train*np.shape(x)[0]))
size_y = int(math.ceil(percent_train*np.shape(y)[0]))
X_train = x[:size_x,:]
X_test = x[size_x:,:]
Y_train = y[:size_y,:]
Y_test = y[size_y:,:]

# Shuffle X_train and Y_train
X_train, Y_train = unison_shuffled_copies(X_train, Y_train)
# Reshape
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

# Load model or fit network
if not os.path.isfile('model.h5'):
    model = get_model()
    EPOCHS = 100
    csv_logger = CSVLogger('training.log', separator=',', append=False)
    model.fit(X_train, Y_train,epochs=EPOCHS,batch_size=128,
                        validation_split=0.2, callbacks=[csv_logger],
                        verbose=1)
    model.save('model.h5')
else:
    model = load_model('model.h5')
    history = pd.read_csv('training.log', sep=',', engine='python')


score, acc = model.evaluate(X_test, Y_test, batch_size=16, verbose=1)

# Predict or load predictions
if not os.path.isfile('yy_predicted_test.npy'):
    yy_predicted = model.predict(X_test)
    yy_predicted_train = model.predict(X_train)
    np.save('yy_predicted_test.npy', yy_predicted)
    np.save('yy_predicted_train.npy', yy_predicted_train)
else:
    yy_predicted = np.load('yy_predicted_test_lstm_64.npy')
    yy_predicted_train = np.load('yy_predicted_train_lstm_64.npy')

# Threshold for predictions
threshold = 0.3
if not os.path.isfile('yy_predicted_test.npy'):
    yy_predicted_thr = yy_predicted > threshold
    yy_predicted_train_thr = yy_predicted_train > threshold
    np.save('yy_predicted_test_threshold.npy', yy_predicted_thr)
    np.save('yy_predicted_train_threshold.npy', yy_predicted_train_thr)
else:
    yy_predicted_thr = np.load('yy_predicted_test_threshold.npy')
    yy_predicted_train_thr = np.load('yy_predicted_train_threshold.npy')


# Evaluation total accuracy
accuracy_test = evalution(Y_test, yy_predicted_thr)

with open('evaluation_test.log', 'a') as file:
    file.write("\nTest -> Accuracy: P peak {:.2f}%, Q peak {:.2f}%, "
        "R peak {:.2f}%, S peak {:.2f}%, T peak {:.2f}%\n".format(
        accuracy_test[0], accuracy_test[1], accuracy_test[2], accuracy_test[3],
        accuracy_test[4]))

accuracy_train = evalution(Y_train, yy_predicted_train_thr)

with open('evaluation_train.log', 'a') as file:
    file.write(("\nTrain -> Accuracy: P peak {:.2f}%, Q peak {:.2f}%, "
        "R peak {:.2f}%, S peak {:.2f}%, T peak {:.2f}%\n".format(
        accuracy_train[0], accuracy_train[1], accuracy_train[2],
        accuracy_train[3], accuracy_train[4])))

# Evaluation range peaks
accuracy_test = evaluation_range_peak(Y_test, yy_predicted_thr)

with open('evaluation_range_test.log', 'a') as file:
    file.write("\nTest -> Accuracy: P peak {:.2f}%, Q peak {:.2f}%, "
        "R peak {:.2f}%, S peak {:.2f}%, T peak {:.2f}%\n".format(
        accuracy_test[0], accuracy_test[1], accuracy_test[2], accuracy_test[3],
        accuracy_test[4]))

accuracy_train = evaluation_range_peak(Y_train, yy_predicted_train_thr)

with open('evaluation_range_train.log', 'a') as file:
    file.write(("\nTrain -> Accuracy: P peak {:.2f}%, Q peak {:.2f}%, "
        "R peak {:.2f}%, S peak {:.2f}%, T peak {:.2f}%\n".format(
        accuracy_train[0], accuracy_train[1], accuracy_train[2],
        accuracy_train[3], accuracy_train[4])))


number_cycles = 3

plot_ecg_data_trained(Y_test, yy_predicted, X_test, number_cycles)
