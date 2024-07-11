import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

def get_prediction(text):
    model = tf.keras.models.load_model("model.keras")
    with open('tokenizer.json', 'r') as f:
        tokenizer = tokenizer_from_json(json.load(f))
    sentence = tokenizer.texts_to_sequences(text)
    sentence = tf.keras.preprocessing.sequence.pad_sequences(sentence, padding='post', maxlen=256)
    prediction = model.predict(sentence)
    if np.argmax(prediction) == 0:
        return "Fake"
    else:
        return "Real"

sentence = "Donald Trump visit's india for the first time"
prediction = get_prediction(sentence)
print(np.argmax(prediction))

demo = gr.Interface(fn=get_prediction, inputs="text", outputs="text")
demo.launch()