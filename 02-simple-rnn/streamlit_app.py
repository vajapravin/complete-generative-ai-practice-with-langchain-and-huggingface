import os
import tempfile
import streamlit as st
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence


st.title('IMDB Sentiment Predictor (Simple RNN)')

st.markdown('Enter a movie review below and the app will predict sentiment using the SimpleRNN model.')

# Load word index
@st.cache_data
def get_word_index():
    return imdb.get_word_index()

word_index = get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}


def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])


def preprocess_text(text, maxlen=500):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=maxlen)
    return padded_review


def try_load_model(path):
    try:
        m = load_model(path)
        return m
    except Exception:
        return None


# Model selection / upload
st.sidebar.header('Model')
model_file_choice = st.sidebar.selectbox('Model source', ('Bundled file (simple_rnn_imdb.h5)', 'Upload .h5'))
model = None

if model_file_choice.startswith('Bundled'):
    # try common locations
    search_paths = [
        'simple_rnn_imdb.h5',
        os.path.join('02-simple-rnn', 'simple_rnn_imdb.h5'),
    ]
    for p in search_paths:
        if os.path.exists(p):
            model = try_load_model(p)
            if model is not None:
                st.sidebar.success(f'Loaded model from {p}')
                break
    if model is None:
        st.sidebar.warning('No bundled model found; switch to Upload .h5 or upload your model below.')

if model_file_choice.startswith('Upload'):
    uploaded = st.sidebar.file_uploader('Upload Keras .h5 model', type=['h5'])
    if uploaded is not None:
        tf_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
        tf_temp.write(uploaded.read())
        tf_temp.flush()
        model = try_load_model(tf_temp.name)
        if model is not None:
            st.sidebar.success('Model uploaded and loaded successfully')
        else:
            st.sidebar.error('Failed to load uploaded model')

if model is None:
    st.warning('Model not loaded yet. Provide a model file or place `simple_rnn_imdb.h5` in the app folder.')


st.subheader('Input review')
default = 'This movie was fantastic! The acting was great and the plot was thrilling.'
user_review = st.text_area('Enter review text', value=default, height=160)

if st.button('Predict'):
    if model is None:
        st.error('No model available to predict. Upload or add a model file first.')
    else:
        pre = preprocess_text(user_review)
        pred = model.predict(pre)
        score = float(pred[0][0])
        sentiment = 'Positive' if score > 0.5 else 'Negative'
        st.write('**Sentiment:**', sentiment)
        st.write('**Score:**', f'{score:.4f}')
        st.write('**Decoded (padded) input:**')
        st.write(pre.tolist())

st.markdown('---')
st.caption('App built from `prediction.ipynb` — uses Keras IMDB word index and a SimpleRNN model.')
