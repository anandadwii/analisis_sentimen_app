import streamlit as st

hide_streamlit_style = """"
<style>
	/* This is to hide hamburger menu completely */
	#MainMenu {visibility: hidden;}
	/* This is to hide Streamlit footer */
	footer {visibility: hidden;}
	/*
	If you did not hide the hamburger menu completely,
	you can use the following styles to control which items on the menu to hide.
	*/
	ul[data-testid=main-menu-list] > li:nth-of-type(4), /* Documentation */
	ul[data-testid=main-menu-list] > li:nth-of-type(5), /* Ask a question */
	ul[data-testid=main-menu-list] > li:nth-of-type(6), /* Report a bug */
	ul[data-testid=main-menu-list] > li:nth-of-type(7), /* Streamlit for Teams */
	ul[data-testid=main-menu-list] > div:nth-of-type(2) /* 2nd divider */
		{display: none;}
</style>

"""


def remove_footer():
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
