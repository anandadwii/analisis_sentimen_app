# allow python to jump read another python file
import os
import sys
import streamlit as st
import pickle
import numpy as np
from keras.utils import pad_sequences
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(1, os.getcwd())
from app.utils.cleaner import *
from app.utils.charts import pie_chart, bar_chart
import app.utils.page_style as ps
seed = 10
np.random.seed(seed)
# start render
st.set_page_config(page_title="Sentimen Analisis", page_icon="🤖")

st.title("Analisis Sentimen")

ps.remove_footer()
st.markdown(
    """
    Aplikasi Website Mini untuk melakukan analisis sentimen dengan dataset sistem tilang elektronik di twitter.
    Model yang digunakan adalah LSTM.
    """
)
raw_text_state = False
if raw_text_state:
    selectbox = st.selectbox("Silahkan pilih", ("", "Raw Text", "csv"))
else:
    selectbox = st.selectbox("Silahkan pilih", ("", "csv"))
model = model_loader()
with open('app/resources/tokenizer/tokenizer.pickle', 'rb') as handler:
    tokenizer = pickle.load(handler)

if selectbox == "Raw Text":
    input_text = st.text_area(label='Sentimen', max_chars=100, height=100)
    trigger = st.button('Analisis!')
    if trigger:
        clean_text = preprocessing_data(input_text)
        sequences = tokenizer.texts_to_sequences([clean_text])
        data = pad_sequences(sequences, maxlen=53)
        # prediction = model.predict(np.array(data, ndmin=2))
        prediction = model.predict([data], verbose=0)
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
            try:
                with st.spinner('Sedang Memproses'):
                    # my_bar = st.progress(0.0, text='sedang membaca csv')
                    df = pd.read_csv(file, sep=';')
                    df['processed'] = df.iloc[:, 0]
                    # my_bar.progress(0.25, 'sedang praproses csv')
                    X = df['processed'].apply(preprocessing_data)
                    # my_bar.progress(0.50, 'sedang tokenisasi')
                    X = tokenizer.texts_to_sequences(X.values)
                    X = pad_sequences(X, maxlen=53)
                    # my_bar.progress(0.75, 'sedang melakukan klasifikasi')
                    y_pred = model.predict(X, verbose=0)
                    y_pred = np.argmax(y_pred, axis=1)
                    df['pred'] = y_pred
                    polarity_encode = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
                    df['pred'] = df['pred'].map(polarity_encode).values
                    max_value = int(df['pred'].value_counts().argmax())
                    # st.write(max_value)
                    label_sentimen = ''
                    if max_value == 2:
                        label_sentimen = 'Positif'
                    elif max_value == 1:
                        label_sentimen = 'Netral'
                    else:
                        label_sentimen = 'Negatif'
                    # my_bar.progress(1.0, 'selesai!')
                    labels = ['Negatif', 'Positif', 'Netral']
                    color_bar = ['red', 'green', 'gray']
                    counts = df['pred'].value_counts()
                    sizes = [counts[0], counts[1], counts[2]]
                    st.subheader(f'Total Sentimen : {df.shape[0]}')
                    st.subheader(f'Sentimen Terbanyak : {label_sentimen}')
                    fig_pie, ax_pie = pie_chart(x=counts.values, label=labels, color=color_bar)
                    fig_bar, ax_bar = bar_chart(x=counts.index, height=counts.values, sizes=sizes, color=color_bar)
                    # pie_image = pie_chart(x=counts.values, label=labels, color=color_bar)
                    # bar_image = bar_chart(x=counts.index, height=counts.values, sizes=sizes,color=color_bar)
                    st.pyplot(fig_pie)
                    st.pyplot(fig_bar)
                    simpan = st.button('Simpan', disabled=True)

            except:
                st.error('File tidak sesuai atau rusak')
