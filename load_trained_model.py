import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from train_model import test_model
from keras.models import load_model
import pickle

# Parameters
# to_load = "nietzsche"
origin = "full_pandas_vocab50"
length = 2000
keep_chars = 50

outfile = open("generated/{}_long.txt".format(origin), "w")
model = load_model("saved_models/{}.h5".format(origin))
char_to_indices = pickle.load(open("saved_models/{}c2i.p".format(origin), "rb"))
indices_to_char = pickle.load(open("saved_models/{}i2c.p".format(origin), "rb"))

for temperature in [0.2, 0.35, 0.5,1]:
    generated_string = test_model(model=model,
                                  char_to_indices=char_to_indices,
                                  indices_to_char=indices_to_char,
                                  seed_string="def ",
                                  temperature=temperature,
                                  test_length=length,
                                  keep_chars=keep_chars)
    output = "Temperature: {} Generated string: {}".format(temperature, generated_string)
    print(output)
    outfile.write(output + "\n")
    outfile.flush()
outfile.close()