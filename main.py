import tensorflow as tf
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop



filepath = tf.keras.utils.get_file('shakespeare.txt', 'http://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

#erading the text to be analysed
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

#selecting a set of range of the text to be analysed
text = text[300000:800000]

#sequential arrangement of the selected text 
characters = sorted(set(text))

#create a dictionary which has the character(c) as the key, index(i) as the value
#enumerate functions asigns one number to each character in the set
index_to_char = dict((i, c) for i, c in enumerate(characters))
char_to_index = dict((c, i) for i, c in enumerate(characters))

#sentences will be target while next_sentense will be features

#specify how long the sentense shall be
SEQ_LENGTH = 40 # we are using 40 char to predict next sentense
STEP_SIZE = 3 #How many characters are we going to shift to the next sentense 

sentenses = []
next_characters = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentenses.append(text[i: i + SEQ_LENGTH])
    next_characters.append(text[i + SEQ_LENGTH])

x = np.zeros((len(sentenses), SEQ_LENGTH, len(characters)), dtype=bool)
y = np.zeros((len(sentenses), len(characters)), dtype=bool)

for i, sentense in enumerate(sentenses):
    for t, character in enumerate(sentense):
        x[i, t, char_to_index[character]] = 1
    y[i, char_to_index[next_characters[i]]] = 1


model = Sequential()
model.add(LSTM(128, input_shape =(SEQ_LENGTH, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer=RMSprop(learning_rate=0.01))

model.fit(x, y, batch_size=256, epochs=4) 

# Saving the trained model
model.save('textgenerator.model.keras')


model =tf.keras.models.load_model('textgenerator.model.keras')


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


#text generation function

def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH -1)
    generated = " "
    sentense = text[start_index: start_index + SEQ_LENGTH]
    generated += sentense
    for i in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, character in enumerate(sentense):
            x[0, t, char_to_index[character]] = 1

        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentense = sentense[1:] + next_character
    return generated




print('---------0.2---------')
print(generate_text(300, 0.2))
print('---------0.4---------')
print(generate_text(300, 0.4))
print('---------0.6---------')
print(generate_text(300, 0.6))
print('---------0.8---------')
print(generate_text(300, 0.8))
print('---------1---------')
print(generate_text(300, 1.0))
