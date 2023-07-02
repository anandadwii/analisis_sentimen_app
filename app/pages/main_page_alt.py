# allow python to jump read another python file
import os
import sys
import traceback

import streamlit as st
import pickle
import numpy as np
from keras.utils import pad_sequences
import pandas as pd
import snscrape.modules.twitter as crawler

sys.path.insert(1, os.getcwd())
from app.utils.cleaner import *
from app.utils.charts import pie_chart, bar_chart
from app.utils.presentations import create_ppt
import app.utils.page_style as ps

seed = 10
np.random.seed(seed)
# start render

st.set_page_config(page_title="Analisis Sentimen", page_icon="🤖")
# ps.remove_footer()
st.title("Analisis Sentimen")
st.markdown(
    """
    Aplikasi Website Mini untuk melakukan analisis sentimen dengan dataset sistem tilang elektronik di twitter.
    Model yang digunakan adalah LSTM.
    """
)

# load model and tokenizer
model = model_loader()
tokenizer = tokenizer_loader()

select_box = st.selectbox("Silahkan pilih", ("", "csv", "live sentiment"))
if select_box == 'csv':
    file = st.file_uploader(label="Unggah CSV dengan separator ;", type="csv", accept_multiple_files=False)
    trigger = st.button('Unggah')
    if trigger and file is not None:
        with st.spinner('Harap Tunggu'):
            try:
                try:
                    df = pd.read_csv(file, header=None, usecols=[0])
                except:
                    pass
                df['processed'] = df[0]
                X = df['processed'].apply(preprocessing_data)
                X = tokenizer.texts_to_sequences(X.values)
                X = pad_sequences(X, maxlen=57)
                y_pred = model.predict(X, verbose=0)
                y_pred = np.argmax(y_pred, axis=1)
                df['pred'] = y_pred
                polarity_encode = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
                df['pred'] = df['pred'].map(polarity_encode).values
                max_value = int(df['pred'].value_counts().argmax())
                label_sentimen = polarity_encode.get(max_value)
                labels = ['Negatif', 'Positif']
                color_bar = ['red', 'green']
                counts = df['pred'].value_counts()
                sizes = [0, 0]
                for key, value in df['pred'].value_counts().items():
                    if key == 'Positif':
                        sizes[1] = value
                    elif key == 'Negatif':
                        sizes[0] = value
                    # elif key == 'Netral':
                    #     sizes[2] = value
                dict_data = {label: size for label, size in zip(labels, sizes)}
                st.write('#')
                st.subheader(f'Total Sentimen : {df.shape[0]}')
                st.write('#')
                highest_sentiment = max(dict_data, key=dict_data.get)
                paragraph = f"Berdasarkan input csv, " \
                            f"sebaran sentimen terbanyak yaitu {highest_sentiment}"
                # st.subheader(f'Sentimen Terbanyak : {label_sentimen}')
                st.write(paragraph)
                st.write('#')
                fig_pie, ax_pie = pie_chart(x=sizes, label=labels, color=color_bar)
                fig_bar, ax_bar = bar_chart(x=labels, height=sizes, sizes=sizes, color=color_bar)
                st.pyplot(fig_pie)
                st.pyplot(fig_bar)

                st.download_button(label="Download as pptx",
                                   data=create_ppt(dict_data, paragraph),
                                   file_name=f"Report CSV sentiment analyzer.pptx")
                st.download_button(
                    label="Download as csv",
                    data=convert_df(df),
                    file_name=f"datasheet.csv",
                    mime="text/csv",
                    key='download-csv'
                )
            except:
                st.error(traceback.print_exc())

elif select_box == 'live sentiment':
    with st.form(key='scrape_form'):
        query_keyword = st.text_input(label='keyword untuk melakukan query')
        start = st.date_input(label='tanggal mulai')
        end = st.date_input(label='tanggal selesai')
        submit_button = st.form_submit_button(label="Submit")
    if submit_button:
        if start == end:
            st.error("tanggal mulai dan tanggal selesai tidak boleh sama")
        if len(query_keyword) == 0:
            st.error("keyword untuk query tidak boleh kosong")
        else:
            query_input = f'{query_keyword} until:{end} since:{start}'
            tweet_list = []
            try:
                with st.spinner('Harap Tunggu'):
                    for i, tweet in enumerate(crawler.TwitterSearchScraper(query_input).get_items()):
                        clean_tweet = preprocessing_data(tweet.content)
                        tweet_list.append(clean_tweet)
                    _result = list(set(tweet_list))
                    df = pd.DataFrame(_result, columns=['processed'])
                    X = df['processed']
                    X = tokenizer.texts_to_sequences(X.values)
                    X = pad_sequences(X, maxlen=57)
                    y_pred = model.predict(X, verbose=0)
                    y_pred = np.argmax(y_pred, axis=1)
                    df['pred'] = y_pred
                    polarity_encode = {0: 'Negatif',
                                       1: 'Netral',
                                       2: 'Positif'}
                    df['pred'] = df['pred'].map(polarity_encode).values
                    df = df[df['pred'] != 'Netral']
                    labels = ['Negatif', 'Positif']
                    color_bar = ['red', 'green']
                    sizes = [0, 0]
                    for key, value in df['pred'].value_counts().items():
                        if key == 'Positif':
                            sizes[1] = value
                        elif key == 'Negatif':
                            sizes[0] = value
                        # elif key == 'Netral':
                        #     sizes[2] = value

                    dict_data = {label: size for label, size in zip(labels, sizes)}
                    st.write('#')
                    st.subheader(f'Total Sentimen : {sum(sizes)}')
                    st.write('#')
                    highest_sentiment = max(dict_data, key=dict_data.get)
                    paragraph = f"Berdasarkan input keyword {query_keyword} mulai {start} hingga {end} menghasilkan " \
                                f"sebaran sentimen terbanyak yaitu {highest_sentiment}"
                    st.write(paragraph)
                    st.write('#')
                    fig_pie, ax_pie = pie_chart(x=sizes, label=labels, color=color_bar)
                    fig_bar, ax_bar = bar_chart(x=labels, height=sizes, sizes=sizes, color=color_bar)
                    st.pyplot(fig_pie)
                    st.pyplot(fig_bar)
                    st.download_button(
                        label="Download as pptx",
                        data=create_ppt(dict_data, paragraph),
                        file_name=f"Report {str(select_box)}.pptx")
                    st.download_button(
                        label="Download as csv",
                        data=convert_df(df),
                        file_name=f"datasheet.csv",
                        mime="text/csv",
                        key='download-csv'
                    )

            except:
                st.error(traceback.print_exc())
