import streamlit as st

st.title("Analisis Sentimen")
st.markdown(
    """
    sentiment analysis app for e-ticket system using LSTM model
    """
)
st.sidebar.markdown("# Main page 🎈")


file = st.file_uploader("upload xlsx or csv file")




