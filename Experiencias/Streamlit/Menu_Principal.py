import streamlit as st
import frontend as fe

# Initialize the session state
if 'page' not in st.session_state:
        st.session_state.page = "main"

# Display the appropriate page
if st.session_state.page == "main":
    
    fe.main_page()

elif st.session_state.page == "pagina1":
    fe.pagina1()
elif st.session_state.page == "pagina2":
        fe.pagina2()