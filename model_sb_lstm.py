import random
import sys
import numpy as np
import operator
import json
import pandas as pd
from keras.models import Sequential, model_from_json, Model
from keras.layers import Input, Lambda, Dense, Flatten, Dropout, Bidirectional, BatchNormalization, Activation, GRU, LSTM
from keras.utils import to_categorical
from keras.losses import binary_crossentropy
from keras import optimizers
from keras import backend as K

import matplotlib.pyplot as plt


# Class determine by value
def value2int_simple(y):
    label = np.copy(y)
    label[y < 0] = 0
    label[y >= 0] = 1
    return label

# Loading 3 miniBatches Together
# Output: prices -> categorical int
def My_Generator(fileName, batch_size):
    chunksize = batch_size
    while True:
        for chunk in pd.read_csv(fileName, chunksize=chunksize, sep=" "):
            batchFeatures = np.array(chunk.iloc[:,:-1])
            batchFeatures = np.reshape(batchFeatures,(batchFeatures.shape[0],30,100))
            batchLabels = np.matrix(chunk.iloc[:,-1]).T
            batchLabels = to_categorical(value2int_simple(batchLabels),num_classes=2).astype("int")
            batchLabels = np.matrix(batchLabels)
            yield batchFeatures,batchLabels

def BiLSTM():
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True) ,input_shape=(30, 100), merge_mode ='ave'))
    model.add(Bidirectional(LSTM(64, return_sequences=True, activation='relu'), merge_mode ='ave'))
    model.add(BatchNormalization(axis=-1))
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.2))
    model.add(Dense(16,activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Dense(2, activation='softmax'))

    adam = optimizers.adam(lr=0.0003)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
    model.summary()
    return model

def train():
    model = BiLSTM()
    # Parameters
    batch_size = 512
    training_filenames = "./input/featureMatrix_0_train.csv"
    validation_filenames = "./input/featureMatrix_0_validation.csv"
    test_filenames = "./input/featureMatrix_0_test.csv"
    num_training_samples = 100000
    num_validation_samples = 25000
    num_test_samples = 25000

    # Training and Validation Set Loader
    my_training_batch_generator = My_Generator(training_filenames, batch_size)
    my_validation_batch_generator = My_Generator(validation_filenames, batch_size)
    my_test_batch_generator = My_Generator(test_filenames, batch_size)
    my_test_batch_generator_one = My_Generator(test_filenames, batch_size)

    print(my_training_batch_generator)

    modelHistory = model.fit_generator(generator=my_training_batch_generator,
                      steps_per_epoch=(int(np.ceil(num_training_samples/ (batch_size)))),
                      epochs=100,
                      verbose=1,
                      validation_data = my_validation_batch_generator,
                      validation_steps = (int(np.ceil(num_validation_samples // (batch_size)))),
                      max_queue_size=32)

    conf = model.evaluate_generator(my_test_batch_generator, steps=(int(np.ceil(num_test_samples/ (batch_size)))),
                             use_multiprocessing=False, verbose=1)
    print(conf)
    print("%s: %.2f%%" % (model.metrics_names[1], conf[1]*100))
    conf = model.predict_generator(my_test_batch_generator_one, steps=(int(np.ceil(num_test_samples/ (batch_size)))), use_multiprocessing=False, verbose=1)
    print(conf)

    model_json = model.to_json()
    with open("./input/model_lstm.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./input/model_lstm.h5")
    print("Saved model to disk")
    print(modelHistory.history.keys())

    # loss
    plt.plot(modelHistory.history['loss'])
    plt.plot(modelHistory.history['val_loss'])
    plt.title('model_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train','validation'],loc='upper left')
    plt.savefig("loss_lstm.png")
    # accuracy
    plt.plot(modelHistory.history['acc'])
    plt.plot(modelHistory.history['val_acc'])
    plt.title('model_accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train','validation'],loc='upper left')
    plt.savefig("accuracy_lstm.png")

def test():
    batch_size = 512
    test_filenames = "./input/featureMatrix_0_test.csv"

    # Load model, weights
    json_file = open('./model/model_lstm.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("./model/model_lstm.h5")
    print("Loaded model from disk")

    num_test_samples = 25000

    # Test Set Loader
    my_test_batch_generator = My_Generator(test_filenames,batch_size)
    my_test_batch_generator_one = My_Generator(test_filenames,batch_size)

    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    conf = loaded_model.evaluate_generator(my_test_batch_generator, steps=(int(np.ceil(num_test_samples/ (batch_size)))),
                             use_multiprocessing=False, verbose=1)
    print(conf)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], conf[1]*100))
    conf = loaded_model.predict_generator(my_test_batch_generator_one, steps=(int(np.ceil(num_test_samples/ (batch_size)))), use_multiprocessing=False, verbose=1)
    print(conf)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "test":
        test()
