import streamlit as st


st.title('Stable Diffusion - Animation')


tab1, tab2 = st.tabs(["Tab 1", "Tab2"])

with tab1:
  col1, col2, col3 = st.columns([2, 3, 2])
    with col1:
      st.radio('Select one:', [1, 2])
