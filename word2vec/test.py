import tensorflow as tf
from tensorflow import keras

settings = {
    'window_size': 2,	    # context window +- center word
	'n': 10,		        # dimensions of word embeddings, also refer to size of hidden layer
	'epochs': 50,		    # number of training epochs
	'learning_rate': 0.01	# learning rate
}


model = keras.Sequential()

model.add(keras.layers.Input(9, input_shape=(9,), activation='relu'))
model.add(keras.layers.Dense(10, activation='relu'))
model.add(keras.layers.Dense(9))

model.compile()

