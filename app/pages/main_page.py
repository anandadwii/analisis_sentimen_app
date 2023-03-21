import streamlit as st

st.header("Analisis Sentimen")
st.markdown(
    """
    sentiment analysis app for e-ticket system using LSTM model
    """
)
# st.sidebar.markdown("# Main page 🎈")

# with st.expander('Text Analysis'):
#     text = st.text_input('Kalimat: ')

with st.expander('Batch Analysis'):
    file = st.file_uploader("Upload CSV file with ; separated")





