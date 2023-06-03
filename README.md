# analisis_sentimen_app
Streamlit for Sentiment analysis App.
Using LSTM deep learning 

## Git
using git version >=24 and dont forget to add at PATH environment table

## initialize python virtual environtment
`python -m venv venv`

## ensure your pip
`python -m ensurepip`

## install requirements dependencies
`pip install -r requirements.txt`

dependecies details:
- snscrape development version, `pip3 install git+https://github.com/JustAnotherArchivist/snscrape.git`
- tensorflow-keras ver >= 2.11
- python-pptx
- matplotlib
- numpy
- streamlit
- emoji
- pandas
- tqdm


## run streamlit
run on terminal with venv
`streamlit run app/pages/main_page.py`

## limit uploaded file for dummies
run on terminal with venv
`streamlit run app/pages/main_page_alt.py --server.maxUploadSize [input your limit]`

