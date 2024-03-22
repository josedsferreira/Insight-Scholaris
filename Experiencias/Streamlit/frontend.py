import streamlit as st
from streamlit_image_select import image_select

def pagina1():
    st.session_state.page = "pagina1"
    st.write('botao um')
    button1 = st.button('Back to main page')
    if button1:
        main_page()

def pagina2():
    st.session_state.page = "pagina2"
    st.write('botao dois')

def main_page():
    st.markdown("""
        <div style='background-color: #F5F5F5; padding: 10px; border-radius: 10px'>
            <h2 style='text-align: center; color: #000'>Welcome to Insight Scholaris!</h2>
            <p style='text-align: center; color: #000'>This is a platform where you can predict, model, manage data and accounts.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style='background-color: #F5F5F5; padding: 10px; border-radius: 10px'>
            <img src='icons/sheet.png' alt='Your Image' style='width: 100%; height: auto;'>
        </div>
        """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    button1 = col1.button('Prever')
    button2 = col2.button('Modelar')
    button3 = col3.button('Dados')
    button4 = col4.button('Gestão de Contas')

    if button1:
        st.session_state.page = "pagina1"
    elif button2:
        st.session_state.page = "pagina2"
    elif button3:
        st.session_state.page = "pagina2"
    elif button4:
        st.session_state.page = "pagina2"

