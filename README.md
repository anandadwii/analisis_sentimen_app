# analisis_sentimen_app
Sentiment analysis application with Streamlit framework and LSTM Deep Learning model.
Model and tokenizer was built at [here](https://github.com/anandadwii/skripsi_bert_ngab).

## initialize python virtual environment
`python -m venv venv`

## ensure your pip
`python -m ensurepip`

## install requirements dependencies

### install snscrape
development version of snscrape:
`pip3 install git+https://github.com/JustAnotherArchivist/snscrape.git`

### install via txt
`pip3 install -r requirements.txt`



## run streamlit
run on terminal with venv
`streamlit run app/pages/main_page_alt.py`

## limit uploaded file
run on terminal with venv
`streamlit run app/pages/main_page_alt.py --server.maxUploadSize [limit in MB]`

