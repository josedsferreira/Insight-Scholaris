from flask import Flask, flash, render_template, request, redirect, url_for, session
from modules import database as mdb
from dotenv import load_dotenv
import os

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
app.debug = False # se true mudanças nos ficheiros automtaticamente recarregam servidor

load_dotenv()
db_name = os.getenv('DB_NAME')

@app.route('/')
def index():
    # check if user is logged in and if not redirect to login page
    if 'loggedin' in session:
        return render_template('index.html', user_type=session['user_type'])
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        print(email, password)

        if not mdb.user_is_valid(database_name=db_name, email=email):
            flash('Email incorreto')
            return redirect(url_for('login'))

        if mdb.is_password_correct(database_name=db_name, email=email, password=password):
            session['loggedin'] = True
            session['email'] = email
            session['user_type'] = mdb.user_type(database_name=db_name, email=email)
            return redirect(url_for('index'))
        
        else:
            flash('Palavra-passe incorreta')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('email', None)
    session.pop('type', None)
    return redirect(url_for('login'))


def main():
    app.run()

if __name__ == '__main__':
    main()