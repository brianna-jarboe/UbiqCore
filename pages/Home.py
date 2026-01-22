import streamlit as st
from util.functions.header import header
from util.functions.struc_3D import show_3D_struc

header(__file__.split('/')[-1].split('.')[0])

col1, col2 = st.columns([1, 1])

with col1:
    st.write(' ')
    st.header('Welcome to UbiqCore!')
    st.markdown('UbiqCore is a protein structure resource created to help advance ubiquitin ligase research. UbiqCore provides ubiquitin-E2-E3 ternary complex structures generated using ColabFold for combinations of known E2 enzymes and RING/U-box E3 ligases. ColabFold models are available for the complexes along with prediction confidence metrics, additional structure analysis data, and final predicted pair probability. Choose from the menu above to browse structure data and/or download .pdb files of ternary complexes of interest. We also provide a listing of experimental structures which contain E2 enzymes and/or E3 ligases available within the Protein Data Bank (PDB). See our paper for full details on the methods used and the data available in UbiqCore.')

with col2:
    show_3D_struc('util/resources/UB-E2-E3.pdb')



