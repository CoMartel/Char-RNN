# char-rnn with keras

char-rnn is a recurrent neural networks for generating texts, characters by characters, based on [Andrej Karpathy's article](http://karpathy.github.io/2015/05/21/rnn-effectiveness)

In this version, I have added support for training on all of the python files of Pandas, but you can change the source file and train on something else.

## Training : 
This is an example call for training : 
```bash
$ python train_model.py --origin full_pandas --rtype LSTM --unit_size 512 --maxlen 240 --num_layers 3
```
## generating text :
The `load_trained_model.py` file can be called to generate some text based on a trained model

There is some examples of generated text in /generated, using a model trained on a mix of pandas and django

## Credits
Keras library example

This is initially based on https://github.com/michaelrzhang/Char-RNN

re-factoring inspired by https://github.com/karpathy/char-rnn and https://github.com/ekzhang/char-rnn-keras
