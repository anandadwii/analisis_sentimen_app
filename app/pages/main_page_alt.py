# allow python to jump read another python file
import os
import sys
sys.path.insert(1, os.getcwd())

import streamlit as st
import pickle
import numpy as np
from keras.utils import pad_sequences
import pandas as pd
import matplotlib.pyplot as plt
from app.utils.cleaner import tokenizing_text
from app.utils.cleaner import cleaning_text
from app.utils.cleaner import filtering_stopwords
from app.utils.cleaner import stemming_text
from app.utils.cleaner import sentence_make
from app.utils.cleaner import model_loader

seed = 10
np.random.seed(seed)
# start render
st.set_page_config(page_title="Sentimen Analisis", page_icon="🤖")

st.title("Analisis Sentimen")
st.markdown(
    """
    Aplikasi Website Mini untuk melakukan analisis sentimen dengan dataset sistem tilang elektronik di twitter.
    Model yang digunakan adalah LSTM.
    """
)

selectbox = st.selectbox("Silahkan pilih", ("", "Raw Text", "csv"))
model = model_loader()
with open('app/resources/tokenizer/tokenizer.pickle', 'rb') as handler:
    tokenizer = pickle.load(handler)

if selectbox == "Raw Text":
    input_text = st.text_area(label='Sentimen', max_chars=100, height=100)
    trigger = st.button('Analisis!')
    if trigger:
        raw_text = cleaning_text(input_text)
        raw_text = tokenizing_text(raw_text)
        # raw_text = filtering_stopwords(raw_text)
        raw_text = stemming_text(raw_text)
        raw_text = sentence_make(raw_text)
        # print(raw_text)
        sequences = tokenizer.texts_to_sequences([raw_text])
        flat_ls = []
        for i in sequences:
            for j in i:
                flat_ls.append(j)
        data = pad_sequences(sequences, maxlen=53)

        # print(data)
        # prediction = model.predict(np.array(data, ndmin=2))
        prediction = model.predict([data], verbose=0)
        print(prediction)
        prediction_prob_negative = prediction[0][0]
        prediction_prob_neutral = prediction[0][1]
        prediction_prob_positive = prediction[0][2]
        prediction_class = prediction.argmax(axis=-1)[0]
        # prediction_class = np.argmax(prediction, axis=1)
        # print(prediction_class )
        # st.header('Prediction using LSTM model')
        if prediction_class == 0:
            st.error(f'Sentimen bernilai Negatif, {prediction_prob_negative}')
        if prediction_class == 1:
            st.info(f'Sentimen bernilai Netral, {prediction_prob_neutral}')
        if prediction_class == 2:
            st.success(f'Sentimen bernilai Postif, {prediction_prob_positive}')


elif selectbox == 'csv':
    file = st.file_uploader(label="Upload csv file dengan separator ;", type="csv", accept_multiple_files=False)
    trigger = st.button('upload')
    if trigger:
        if file is not None:
            with st.spinner('Loading'):
                df = pd.read_csv(file, sep=';')
                df['processed'] = df.iloc[:, 0]
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
