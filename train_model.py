
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM,GRU, Embedding
from keras.layers.wrappers import TimeDistributed
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint,EarlyStopping

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import numpy as np
import random
import pickle
import sys
import argparse
import os
import gc



# Loosely follows Karparthy, Keras library example, and mineshmathew's repo
# I added code and comments for clarity and ease of use.


def sample(preds, temperature=1.0):
    """
    helper function to sample an index from a probability array (from Keras library)
    increasing the temperature flttens the characters probability distribution, and allow for more randomness
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def test_model(model, char_to_indices, indices_to_char, seed_string="def ", temperature=1.0, test_length=150,keep_chars=50):
    """
    Higher temperatures correspond to more potentially creative sentences (at the cost of mistakes)
    keep_chars correspond to the number of characters the network will keep as input.
    This increase the testing speed, but a number too low can decrease performance
    """
    num_chars = len(char_to_indices.keys())
    for i in range(test_length):
        test_in = np.zeros((1, len(seed_string[-keep_chars:])))
        for t, char in enumerate(seed_string[-keep_chars:]):
            test_in[0, t] = char_to_indices[char]
        # input 'goodby', desired output is 'oodbye' # possible todo: show that this holds for the model
        entire_prediction = model.predict(test_in, verbose=0)[0]
        next_index = sample(entire_prediction[-1], temperature)
        next_char = indices_to_char[next_index]
        seed_string = seed_string + next_char
    return seed_string

def build_model(unit_size,
                num_chars,
                maxlen,
                batch_size,
                vocab_size,
                num_layers,
                dropout,
                rtype ='LSTM'):
    model = Sequential()
    model.add(Embedding(num_chars, vocab_size))
    for i in range(num_layers):
        if rtype == 'LSTM':
            model.add(LSTM(unit_size, return_sequences=True))
        elif rtype == 'GRU':
            model.add(GRU(unit_size, return_sequences=True))
        else:
            raise NotImplementedError('Choose rtype between LSTM and GRU')
        model.add(Dropout(dropout))

    model.add(TimeDistributed(Dense(num_chars)))
    model.add(Activation('softmax'))
    return model

def plot_history(history,model_name):
    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./figures/{}_acc_history.png'.format(model_name), bbox_inches='tight')
    plt.close()
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./figures/{}_loss_history.png'.format(model_name), bbox_inches='tight')
    plt.close()


if __name__ == "__main__":

### Parameters

    origin = "full_pandas_with_import" # select the original source
    seed = 2 # random seed

    # Recurrent unit parameters 
    rtype ='LSTM' # also accepted : 'LSTM', 'GRU'
    unit_size = 512  
    num_layers = 3
    dropout = 0
    batch_size = 512
#     vocab_size = 200

    # optimization parameters
    optimizer = 'rmsprop'
    training_epochs = 50
    patience = 2
    test_size = 0.25

    # how we break sentences up
    maxlen = 240 # perhaps better for step not to divide maxlen (to get more overlap) 
    step = 13

    # testing
    testing = False # set to True if you want to run a test with the trained model
    test_length = 500
    keep_chars = 50
    
### Setting argument with arg parser. If not specified, take default values above
    parser = argparse.ArgumentParser(description='Train the model on some text.')
    parser.add_argument('--origin', default=origin,
                        help='name of the text file to train from')
    parser.add_argument('--rtype', default=rtype,
                        help='type of recurrent cell : LSTM or GRU')
    parser.add_argument('--unit_size', type=int, default=unit_size,
                        help='number of units')
    parser.add_argument('--maxlen', type=int, default=maxlen,
                        help='maximum length of sentence')
    parser.add_argument('--num_layers', type=int, default=num_layers,
                        help='number of layers')
    args = parser.parse_args()
    
    origin = args.origin
    rtype = args.rtype
    unit_size = args.unit_size
    maxlen =  args.maxlen
    num_layers = args.num_layers
    
    model_name = "{}_usize{}_maxlen{}_numlayers{}_dropout{}".format(origin,unit_size,maxlen,num_layers,dropout)

    if not os.path.exists("./saved_models/{}/".format(model_name)): # create new folder for the model
        os.makedirs("./saved_models/{}/".format(model_name))

    print("Model_name : {}".format(model_name))
    
### Select source
    if "nietzsche" in origin:
        path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
        text = open(path).read().lower()
    elif "obama" in origin:
        text = open("data/obama.txt").read().lower()
    elif "sonnet" in origin:
        text = open("data/sonnet.txt").read().lower() 
    elif "paulgraham" in origin:
        text = open("data/paulgraham.txt").read().lower()
    elif "python" in origin:
        text = open("data/generic.py").read().lower()
    elif "full_pandas" in origin:
        text = open("data/full_pandas.py",encoding="utf-8").read().lower()
    else:  # add your own text! (something to do with sports would be interesting)
        raise NotImplementedError

    np.random.seed(seed)
    random.seed(seed)

### reading corpus
    print('corpus length:', len(text))
    chars = sorted(list(set(text)))
    num_chars = len(chars)
    print('characters: ', chars)
    print('total characters in vocabulary:', num_chars)
    vocab_size = num_chars

### dictionaries to convert characters to numbers and vice-versa
    try : 
        char_to_indices = pickle.load(open("saved_models/{}_c2i.p".format(origin), "rb"))
        indices_to_char = pickle.load(open("saved_models/{}_i2c.p".format(origin), "rb"))
        print("loading char_to_indices")
    except Exception as e:
        print('Not able to load char_to_indice : {}'.format(e))
        char_to_indices = dict((c, i) for i, c in enumerate(chars))
        indices_to_char = dict((i, c) for i, c in enumerate(chars))
        pickle.dump(char_to_indices, open("saved_models/{}_c2i.p".format(origin), "wb"))
        pickle.dump(indices_to_char, open("saved_models/{}_i2c.p".format(origin), "wb"))

### cut the text in semi-redundant sequences of maxlen characters 
    sentences = []
    targets = []
    idx_text = [char_to_indices[c] for c in text]
    for i in range(0, len(idx_text) - maxlen - 1, step):
        sentences.append(idx_text[i: i + maxlen])
        targets.append(idx_text[i + 1: i + maxlen + 1])
    
    sentences = np.array(sentences)
    sentences = sentences[:batch_size*(len(sentences)//batch_size)] # keep same number of sentences for each batch
    print('number of sequences:', len(sentences))

    y = np.zeros((len(sentences), maxlen, num_chars), dtype=np.bool) # target must be vectorized to match the output dimensions
    for i in range(len(sentences)):
        for j in range(maxlen):
            y[i][j][targets[i][j]] = 1

### split in train - test
    train_sentences, val_sentences, train_targets, val_targets = train_test_split(sentences, y, test_size=test_size, random_state=118)
    
### release some memory
    del targets, text, idx_text,sentences, y
    gc.collect()       
    
##### start building model
    print('Building model...')
#     model = load_model("saved_models/full_pandas_with_import_usize512_maxlen240_numlayers3_dropout0/weights-02-0.82.h5")
     
    model = build_model(unit_size,
                        num_chars,
                        maxlen,
                        batch_size,
                        vocab_size,
                        num_layers,
                        dropout,
                        rtype = rtype)
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
              
    print(model.summary())
    
    filepath="saved_models/{}/".format(model_name)
    filepath+="weights-{epoch:02d}-{val_acc:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_acc',min_delta=0,patience=patience,verbose=0, mode='auto')
    callbacks_list = [checkpoint,early_stopping]
    print('...model built!')


    
### training
    history = model.fit(train_sentences,
                        train_targets,
                        batch_size=batch_size,
                        epochs=training_epochs,
                        validation_data = (val_sentences,val_targets),
                        callbacks=callbacks_list,
                        verbose=1)
    
    model.save("saved_models/{}/final.h5".format(model_name))
    
    plot_history(history,model_name)

    if testing:   
        with open("generated/{}.txt".format(model_name), "w",encoding="utf-8") as outfile :
            for temperature in [0.35,1]:
                generated_string = test_model(model,
                                              char_to_indices=char_to_indices,
                                              indices_to_char=indices_to_char,
                                              temperature=temperature,
                                              test_length=test_length,
                                              keep_chars=keep_chars)
                try:
                    output = "Temperature: {}, generated string:\n{}".format(temperature, generated_string)
                    print(output)
                    outfile.write(output + "\n")
                    outfile.flush()
                except Exception as e :
                    print(e)
                    # if something goes wrong in the output
                    pass