import emoji
import re
import numpy as np
from keras.utils import pad_sequences
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import streamlit as st
from keras.models import load_model

# nltk.download('stopwords')
# nltk.download('punkt')
import pickle


def remove_emoji():
    """menghilangkan emoji"""
    emojis = sorted(emoji.EMOJI_DATA,
                    key=len, reverse=True)
    pattern = u'(' + u'|'.join(
        re.escape(u) for u in emojis) + u')'
    return re.compile(pattern)


def tokenizing_text(value):
    """
    tokenize kalimat menjadi list kata
    """
    result = word_tokenize(value)
    return result


def cleaning_text(value):
    result = value.lower().strip()
    # result = remove_three_same_char(result)
    result = ' '.join(result.split())
    result = re.sub(r'(@|https?)\S+|#[A-Za-z0-9_]+',
                    '', result).replace("&amp;", "dan")
    result = re.sub(r'RT[\s]+', '', result)
    result = begone_emoji.sub(repl='', string=result)
    result = re.sub(r'[0-9]+', '', result)
    result = result.replace('\n', ' ')
    result = result.translate(
        str.maketrans('', '', string.punctuation))
    return result


def filtering_stopwords(value):
    """ menghilangkan stopword pada kalimat"""
    list_stopwords = set(
        stopwords.words('indonesian'))
    list_stopwords.remove("tidak")
    filtered = []
    for text in value:
        if text not in list_stopwords:
            filtered.append(text)
    return filtered


def stemming_text(value):
    """ mengubah kata menjadi kata dasar untuk menyeragamkan kata"""
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return [stemmer.stem(word) for word in value]


def sentence_make(tokenized):
    """ menggambungkan list kata menjadi string"""
    return ' '.join(word for word in tokenized)


def preprocessing_data(var, stopword: bool = False):
    """ menggabungkan seluruh function untuk melakukan preprocessing data"""
    bucket = cleaning_text(var)
    bucket = tokenizing_text(bucket)
    bucket = filtering_stopwords(bucket)
    bucket = stemming_text(bucket)
    bucket = sentence_make(bucket)
    return bucket


@st.cache_resource
def model_loader():
    """ fungsi untuk load model ke dalam streamlit framework"""
    return load_model('app/resources/model/balanced_manual_model.h5')


@st.cache_resource
def tokenizer_loader():
    """fungsi untuk load tokenizer ke dalam cache website"""
    with open('app/resources/tokenizer/manual_tokenizer.pickle', 'rb') as handler:
        tokenizer = pickle.load(handler)
    return tokenizer


begone_emoji = remove_emoji()

@st.cache_resource
def convert_df(df):
    return df.to_csv(index=False, sep=';', header=True, index_label=None)
