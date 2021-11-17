# TODO: Fix git lfs
#   Pull the changes that were not pulled inside this branch [+]
#   push the stuff [+]
#   Test the model that is trained if it does it's job!!! []
#   Write my professor about the progress! []
import os

import tensorflow as tf
import transformers

# load the model
path_to_model = os.path.join(os.curdir, 'saved_model')
model = tf.keras.models.load_model(path_to_model)

# load the tokenizer
path_to_tokenizer = os.path.join(os.curdir, 'bert-base-german-cased')
tokenizer = transformers.BertTokenizer.from_pretrained(path_to_tokenizer)
text_to_test = 'Geburtsort'
inp = tokenizer(text_to_test)
print(inp)
# inp['input_ids'] = inp['input_ids']

# Test the Model
out = model(inp)
out_tokens = tokenizer.convert_ids_to_tokens(out)
print(out_tokens)
out_text = tokenizer.convert_tokens_to_string(out_tokens)
print(out_text)