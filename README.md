# analisis_sentimen_app
Streamlit for Sentiment analysis App.
Using LSTM deep learning 

## initialize python virtual environtment
python -m venv venv

## ensure your pip
python -m ensurepip

## install requirements dependencies
pip install -r requirements.txt


## run streamlit
streamlit run app/pages/main_page.py

## limit uploaded file for dummies
streamlit run app/pages/main_page_alt.py --server.maxUploadSize [input your limit]

