import streamlit as st
import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import random
import pickle
from streamlit_chat import message as st_message


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

st.title("Diega, Le Wagon Web Assistant")



#tag = lbl_encoder.inverse_transform([np.argmax(result)])
if "history" not in st.session_state:
    st.session_state.history = []

def generate_answer():
    #tokenizer, model = get_models()


    user_message = st.session_state.input_text



    #inputs = tokenizer(st.session_state.input_text, return_tensors="pt")

    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([user_message]),
                                             truncating='post', maxlen=max_len))


    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    for i in data['intents']:
            if i['tag'] == tag:
                st.session_state.history.append({"message": np.random.choice(i['responses']), "is_user": False})
                st.session_state.history.append({"message": user_message, "is_user": True})

                #print (np.random.choice(i['responses']))

    #message_bot = tokenizer.decode(
    #    result[0], skip_special_tokens=True
    #)  # .replace("<s>", "").replace("</s>", "")





st.text_input("Type in your questions below: ", key="input_text", on_change=generate_answer)

for chat in st.session_state.history:
    st_message(**chat)  # unpacking
