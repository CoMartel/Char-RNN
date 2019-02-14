import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from train_model import test_model
from keras.models import load_model
import pickle

# Parameters
origin = "full_pandas_with_import"
model_name = "full_pandas_with_import_usize512_maxlen240_numlayers3_dropout0"
weights = "weights-13-0.94.h5"
length = 10000
keep_chars = 50


model = load_model("saved_models/{}/{}".format(model_name,weights))
char_to_indices = pickle.load(open("saved_models/{}_c2i.p".format(origin), "rb"))
indices_to_char = pickle.load(open("saved_models/{}_i2c.p".format(origin), "rb"))

for temperature in [0.1,0.35,0.5,1,2]:
    generated_string = test_model(model=model,
                                  char_to_indices=char_to_indices,
                                  indices_to_char=indices_to_char,
                                  seed_string="def ",
                                  temperature=temperature,
                                  test_length=length,
                                  keep_chars=keep_chars)
    
    with open("generated/{}.txt".format(model_name), "a",encoding="utf-8") as outfile:
        output = "Temperature: {} Generated string: \n{}\n".format(temperature, generated_string)
        print(output)
        outfile.write(output + "\n")