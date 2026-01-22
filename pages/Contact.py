import streamlit as st
from util.functions.header import header

header(__file__.split('/')[-1].split('.')[0])

st.header('Contact', divider='grey')

st.write("")

st.write("Developed and Maintained by Brianna Jarboe and Roland Dunbrack")
# link to Dunbrack lab
st.write("[Dunbrack Lab](https://dunbrack.fccc.edu/)")

st.write("")
# Brianna
st.write("**Brianna Jarboe**")
st.write("Email: jarboebrianna@gmail.com")

st.write("")
# Roland
st.write("**Roland Dunbrack**")
st.write("Email: roland.dunbrack@fccc.edu")

st.write("")
st.write("")
