import streamlit as st
from keras.models import load_model
import pickle
import string
import re
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import nltk
import numpy as np
from keras.utils import pad_sequences
import pandas as pd
import emoji
import matplotlib.pyplot as plt

# nltk.download('stopwords')
# nltk.download('punkt')
seed = 10
np.random.seed(seed)

# render
st.set_page_config(page_title="Sentimen Analisis", page_icon="🤖")
if "sentiment" not in st.session_state:
    st.session_state.summary = ""
if "error" not in st.session_state:
    st.session_state.error = ""


def remove_emoji():
    """menghilangkan emoji"""
    emojis = sorted(emoji.EMOJI_DATA, key=len, reverse=True)
    pattern = u'(' + u'|'.join(re.escape(u) for u in emojis) + u')'
    return re.compile(pattern)


begone_emoji = remove_emoji()


def cleaning_text(value):
    result = value.lower().strip()
    # result = remove_three_same_char(result)
    result = ' '.join(result.split())
    result = re.sub(r'(@|https?)\S+|#[A-Za-z0-9_]+', '', result).replace("&amp;", "dan")
    result = re.sub(r'RT[\s]+', '', result)
    # result = begone_emoji.sub(repl='', string=result)
    result = re.sub(r'[0-9]+', '', result)
    result = result.replace('\n', ' ')
    result = result.translate(str.maketrans('', '', string.punctuation))
    return result


def tokenizing_text(value):
    result = word_tokenize(value)
    return result


def filtering_stopwords(value):
    list_stopwords = set(stopwords.words('indonesian'))
    filtered = []
    for text in value:
        if text not in list_stopwords:
            filtered.append(text)
    return filtered


def stemming_text(value):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return [stemmer.stem(word) for word in value]


def sentence_make(tokenized):
    return ' '.join(word for word in tokenized)


st.title("Analisis Sentimen")
st.markdown(
    """
    Aplikasi Website Mini untuk melakukan analisis sentimen dengan dataset sistem tilang elektronik di twitter.
    """
)

selectbox = st.selectbox("Raw text or csv source", ("", "Raw Text", "csv"))
model = load_model('app/resources/model/model.h5')
with open('app/resources/tokenizer/tokenizer.pickle', 'rb') as handler:
    tokenizer = pickle.load(handler)

if selectbox == "Raw Text":
    # raw_text = st.text_input(label="Sentimen:", max_chars=100)
    input_text = st.text_area(label='Sentimen', max_chars=100, height=100)
    trigger = st.button('predict!')
    if trigger:
        # tokenize_text = raw_text
        raw_text = cleaning_text(input_text)
        raw_text = tokenizing_text(raw_text)
        # raw_text = filtering_stopwords(raw_text)
        raw_text = stemming_text(raw_text)
        raw_text = sentence_make(raw_text)
        print(raw_text)
        sequences = tokenizer.texts_to_sequences([raw_text])
        flat_ls = []
        for i in sequences:
            for j in i:
                flat_ls.append(j)
        data = pad_sequences(sequences, maxlen=53)

        print(data)
        # prediction = model.predict(np.array(data, ndmin=2))
        prediction = model.predict([data], verbose=0)
        print(prediction)
        prediction_prob_negative = prediction[0][0]
        prediction_prob_neutral = prediction[0][1]
        prediction_prob_positive = prediction[0][2]
        prediction_class = prediction.argmax(axis=-1)[0]
        # prediction_class = np.argmax(prediction, axis=1)
        kontol = np.argmax(prediction, axis=1)
        # print(prediction_class )
        # st.header('Prediction using LSTM model')
        if prediction_class == 0:
            st.error('Sentimen bernilai negatif')
        if prediction_class == 1:
            st.info('Sentimen bernilai netral')
        if prediction_class == 2:
            st.success('Sentimen bernilai postif')


elif selectbox == 'csv':
    file = st.file_uploader(label="Upload csv file dengan separator ;", type="csv", accept_multiple_files=False)
    trigger = st.button('upload')
    if trigger:
        if file is not None:
            with st.spinner('Loading'):
                df = pd.read_csv(file, sep=';')
                # st.write(f'Total Sentimen : {df.shape[0]}')
                # print(df.iloc[:, 0])
                df['processed'] = df.iloc[:, 0]
                # print(df['processed'])
                df['processed'] = df['processed'].apply(cleaning_text)
                df['processed'] = df['processed'].apply(tokenizing_text)
                df['processed'] = df['processed'].apply(stemming_text)
                X = df['processed'].apply(sentence_make)
                X = tokenizer.texts_to_sequences(X.values)
                X = pad_sequences(X, maxlen=53)
                y_pred = model.predict(X, verbose=0)
                y_pred = np.argmax(y_pred, axis=1)
                df['pred'] = y_pred
                polarity_encode = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
                df['pred'] = df['pred'].map(polarity_encode).values

                st.write(f'Total Sentimen : {df.shape[0]}')
                max_value = int(df['pred'].value_counts().argmax())
                # st.write(max_value)
                label_sentimen = ''
                if max_value == 2:
                    label_sentimen = 'Positif'
                elif max_value == 1:
                    label_sentimen = 'Netral'
                else:
                    label_sentimen = 'Negatif'
                labels = ['Negatif', 'Positif', 'Netral']
                color_bar = ['red', 'green', 'gray']
                counts = df['pred'].value_counts()
                fig, ax = plt.subplots(figsize=(6, 6))
                # sizes = [row for row in df['pred'].value_counts()]
                sizes = [counts[0], counts[1], counts[2]]
                label = list(df['pred'].value_counts().index)
                explode = (0.1, 0, 0)
                ax.pie(x=counts.values, labels=labels, colors=color_bar,
                       autopct='%1.1f%%',
                       textprops={'fontsize': 11})
                ax.set_title('sentimen terhadap sistem tilang elektronik di twitter', fontsize=12)

                fig_bar, ax_bar = plt.subplots()
                ax_bar.bar(counts.index, counts.values, color=color_bar)
                for i in range(len(counts.values)):
                    ax_bar.text(i, sizes[i], sizes[i], ha='center')
                ax_bar.set_xlabel('Nilai Prediksi')
                ax_bar.set_ylabel('Jumlah')
                ax_bar.set_title('Diagram Batang Prediksi')
                # legend = ax_bar.legend(labels=labels, loc='best')
                st.write(f'Sentimen Terbanyak : {label_sentimen}')
                st.pyplot(fig)
                st.pyplot(fig_bar)
                simpan = st.button('Simpan', disabled=True)

