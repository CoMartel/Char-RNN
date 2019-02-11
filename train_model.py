
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM,GRU, Embedding
from keras.layers.wrappers import TimeDistributed
from keras.utils.data_utils import get_file
import numpy as np
import random
import pickle
import sys
import argparse


# Loosely follows Karparthy, Keras library example, and mineshmathew's repo
# I added code and comments for clarity and ease of use.


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array (from Keras library)
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature  # Taking the log should be optional? add fudge factor to avoid log(0)
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
    model.add(Embedding(num_chars, vocab_size)) #,batch_input_shape=(batch_size, maxlen)))
#                         ,batch_input_shape=(128, 120)))
#                         input_length=maxlen))
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


if __name__ == "__main__":

    # Parameters

    # Random seed. Change to get different training results / speeds
    # origin = "obama2"  # used to name files saved as well
    # origin = "nietzsche"
    origin = "full_pandas"
    seed = 2

    # Recurrent unit parameters 
    rtype ='LSTM' # also accepted : 'LSTM', 'GRU'
    unit_size = 512  # can increase more if using dropout
    num_layers = 3
    dropout = 0.2
    batch_size = 512
#     vocab_size = 200

    # optimization parameters
    optimizer = 'rmsprop'
    training_epochs = 3
    epoch_steps = 1

    # how we break sentences up
    maxlen = 120 # perhaps better for step not to divide maxlen (to get more overlap) 
    step = 13
    # increasing maxlen should allow for more coherent thoughts
    # previously maxlen = 40, step = 10 before

    # testing
    test_length = 500
    keep_chars = 50
    
    parser = argparse.ArgumentParser(description='Train the model on some text.')
    parser.add_argument('--origin', default=origin,
                        help='name of the text file to train from')
    parser.add_argument('--unit_size', type=int, default=unit_size,
                        help='number of units')
    parser.add_argument('--maxlen', type=int, default=maxlen,
                        help='maximum length of sentence')
#     parser.add_argument('--resume', action='store_true',
#                         help='resume from previously interrupted training')
    args = parser.parse_args()
    
    origin = args.origin
    unit_size = args.unit_size
    maxlen =  args.maxlen
    
    print('origin : {}, unit_size : {}, maxlen : {}'.format(origin,unit_size,maxlen))
    
    # Select source
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

    print('corpus length:', len(text))
    chars = sorted(list(set(text)))
    num_chars = len(chars)
    print('characters: ', chars)
    print('total characters in vocabulary:', num_chars)
    vocab_size = num_chars

    # dictionaries to convert characters to numbers and vice-versa
    try : 
        char_to_indices = pickle.load(open("saved_models/{}c2i.p".format(origin), "rb"))
        indices_to_char = pickle.load(open("saved_models/{}i2c.p".format(origin), "rb"))
        print("loading char_to_indices")
    except Exception as e:
        print('Not able to load char_to_indice : {}'.format(e))
        char_to_indices = dict((c, i) for i, c in enumerate(chars))
        indices_to_char = dict((i, c) for i, c in enumerate(chars))
        pickle.dump(char_to_indices, open("saved_models/{}c2i.p".format(origin), "wb"))
        pickle.dump(indices_to_char, open("saved_models/{}i2c.p".format(origin), "wb"))

    # cut the text in semi-redundant sequences of maxlen characters 
    ########### get the sentences
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
        target = targets[i]
        for j in range(maxlen):
            y[i][j][target[j]] = 1
    targets = y 

              
##### start building model
    print('Building model...')
    
    try : 
        model = load_model("saved_models/{}_usize{}_maxlen{}.h5".format(origin,unit_size,maxlen))
        print('Loading existing model')
    except Exception as e:
        print('Not able to load model : {}'.format(e))        
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
    print('...model built!')

    
    
# saves generated text in file
    with open("generated/{}_usize{}_maxlen{}.txt".format(origin,unit_size,maxlen), "w") as outfile :
        # -----Training-----
        for i in range(0,training_epochs,epoch_steps):
            history = model.fit(sentences, targets, batch_size=batch_size, epochs=epoch_steps, verbose=1)

            print('-' * 10 + ' Iteration: {} '.format(i) + '-' * 10)
            outfile.write("\n" + '-' * 10 + ' Iteration: {} '.format(i) + '-' * 10 + "\n")
            
            print('loss is {}'.format(history.history['loss'][0]))
            outfile.write('loss is {}'.format(history.history['loss'][0])+ "\n")
            
            if i>0:
                for temperature in [0.35]:
                    generated_string = test_model(model,
                                                  char_to_indices=char_to_indices,
                                                  indices_to_char=indices_to_char,
                                                  temperature=temperature,
                                                  test_length=test_length,
                                                  keep_chars=keep_chars)
                    output = "Temperature: {}, generated string:\n{}".format(temperature, generated_string)
                    print(output)
                    outfile.write(output + "\n")
                    outfile.flush()


    model.save("saved_models/{}_usize{}_maxlen{}.h5".format(origin,unit_size,maxlen))
