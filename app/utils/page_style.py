import streamlit as st
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """


def remove_footer():
    """ fungsi untuk menghilangkan footer made using streamlit"""
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
