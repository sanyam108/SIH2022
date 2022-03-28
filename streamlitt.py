import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components

st.title('toTransliterate ')

# Using object notation
add_selectbox = st.sidebar.selectbox(
    "Choose the mode of Input ",
    ("Text", "Audio", "Image")
)

# Using "with" notation
# with st.sidebar:
#     add_radio = st.radio(
#         "Choose the mode of Input",
#         ("Text", "Audio","Image")
#     )


st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://unsplash.com/photos/6QnEf_b47eA")
    }
   .sidebar .sidebar-content {
         background-image: linear-gradient(#FFFFFF,#FFFFFF);
         color: white;
    }
    body {
    background-image: url("https://i.ibb.co/Sny7Wzm/Mid-shade.png");
    background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)
import pickle

# loading
with open("C:/Users/sanyam ahuja/Downloads/tokenizer.pickle", 'rb') as handle:
    tamil_tokenizer = pickle.load(handle)
with open("C:/Users/sanyam ahuja/Downloads/tokenizer2.pickle", 'rb') as handle:
    english_tokenizer = pickle.load(handle)
from tensorflow import keras
model = keras.models.load_model("C:/Users/sanyam ahuja/Downloads/translation.h5")
a = []
t = [0] * 74
s = [0] * 74
    
token=tamil_tokenizer.word_index['அரசியல்']
t[0] = token
a.append(t)
a.append(s)
a=np.array(a)
def logits_to_text(logits, tokenizer):
    
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])
print(logits_to_text(model.predict(a[:1])[0], english_tokenizer))

components.html(
    """
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>

    <meta charset="utf-8" />
    <title>Semantic UI CDN</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/semantic-ui/1.11.8/semantic.min.css"/>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.1.3/jquery.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/semantic-ui/1.11.8/semantic.min.js"></script>

   
  
      <div class="ui segment">
                <div class="ui two column very relaxed grid">
                  <div class="column">
                    <select class="ui dropdown">
                        <option value="">Select Language</option>
                        <option value="1">Bengali</option>
                        <option value="0">Marathi</option>
                        <option value="2">Tamil</option>
                    </select>
                    <br>
                    <div class="ui form">
                        <div class="field">
                          <label></label>
                          <textarea placeholder="Enter Text.."></textarea>
                        </div>
                      </div>
                  </div>
                  <div class="column">
                    <select class="ui dropdown">
                        <option value="">Select Language</option>
                        <option value="1">English</option>
                    </select>
                    <div class="ui form">
                        <div class="field">
                          <label></label>
                          <textarea placeholder="Translated.."></textarea>
                        </div>
                      </div>
                  </div>
                </div>
                <div class="ui vertical divider">
                  TO
                </div>
              </div>
   
    """,
    height=250,
)