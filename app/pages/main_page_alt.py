import streamlit as st

# render
st.set_page_config(page_title="Sentimen Analisis", page_icon="🤖")
if "sentiment" not in st.session_state:
    st.session_state.summary = ""
if "error" not in st.session_state:
    st.session_state.error = ""

st.title("Analisis Sentimen")
st.markdown(
    """
    Aplikasi Website Mini untuk melakukan analisis sentimen dengan dataset sistem tilang elektronik di twitter.
    """
)

selectbox = st.selectbox("Raw text or csv source", ("", "Raw Text", "csv"))

if selectbox == "Raw Text":
    raw_text = st.text_input(label="Sentimen:", max_chars=100)

elif selectbox == 'csv':
    file = st.file_uploader(label="Upload csv file dengan separator ;", type="csv", accept_multiple_files=False)
