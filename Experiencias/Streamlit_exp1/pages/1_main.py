import streamlit as st
import streamlit_authenticator as stauth

names = ['John Smith','Rebecca Briggs']
usernames = ['jsmith','rbriggs']
passwords = ['123','456']

hashed_passwords = stauth.Hasher(passwords).generate()

config = {
    'John Smith': {'username': 'jsmith', 'password': hashed_passwords[0]},
    'Rebecca Briggs': {'username': 'rbriggs', 'password': hashed_passwords[1]}
}

authenticator = stauth.Authenticate(config, \
                                    'some_cookie_name','some_signature_key', cookie_expiry_days=30)

name, authentication_status, username = authenticator.login('Login','main')

if st.session_state['authentication_status']:
    authenticator.logout('Logout', 'main')
    st.write(f'Welcome *{st.session_state["name"]}*')
    st.title('Some content')
elif st.session_state['authentication_status'] == False:
    st.error('Username/password is incorrect')
elif st.session_state['authentication_status'] == None:
 st.warning('Please enter your username and password')