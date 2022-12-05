import streamlit as st
import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import random
import pickle

with open("intents.json") as file:
    data = json.load(file)

model = keras.models.load_model('chat_model')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# load label encoder object
with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

# parameters
max_len = 20


#from streamlit_chat import message as st_message
#from transformers import BlenderbotTokenizer
#from transformers import BlenderbotForConditionalGeneration


#@st.experimental_singleton
# def get_models():
#     # it may be necessary for other frameworks to cache the model
#     # seems pytorch keeps an internal state of the conversation
#     model_name = "./chat_model"
#     tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
#     model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
#     return tokenizer, model


# if "history" not in st.session_state:
#     st.session_state.history = []

st.title("Hello Chatbot")



#tag = lbl_encoder.inverse_transform([np.argmax(result)])


def generate_answer():
    #tokenizer, model = get_models()


    user_message = st.session_state.input_text



    #inputs = tokenizer(st.session_state.input_text, return_tensors="pt")

    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inputs]),
                                             truncating='post', maxlen=max_len))


    #message_bot = tokenizer.decode(
    #    result[0], skip_special_tokens=True
    #)  # .replace("<s>", "").replace("</s>", "")

    #st.session_state.history.append({"message": user_message, "is_user": True})
    #st.session_state.history.append({"message": message_bot, "is_user": False})


st.text_input("Talk to the bot", key="input_text", on_change=generate_answer)

#for chat in st.session_state.history:
#    st_message(**chat)  # unpacking
