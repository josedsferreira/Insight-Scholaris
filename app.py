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
		return render_template('index.html', user_type=session['user_type'], default_pw=session['default_pw'])
	return redirect(url_for('login'))

# ============ LOGIN / LOGOUNT ============
@app.route('/login', methods=['GET', 'POST'])
def login():
	if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
		email = request.form['email']
		password = request.form['password']

		if not mdb.user_is_valid(database_name=db_name, email=email):
			flash('Email incorreto', 'error')
			return redirect(url_for('login'))

		if mdb.is_password_correct(database_name=db_name, email=email, password=password):
			session['loggedin'] = True
			session['email'] = email
			session['user_type'], session['default_pw'] = mdb.user_info(database_name=db_name, email=email)
			return redirect(url_for('index'))
		
		else:
			flash('Palavra-passe incorreta', 'error')
			return redirect(url_for('login'))
	
	return render_template('login.html')

@app.route('/logout')
def logout():
	session.pop('loggedin', None)
	session.pop('email', None)
	session.pop('type', None)
	session.pop('default_pw', None)
	return redirect(url_for('login'))

# ============ MENU CONTA ============
@app.route("/account")
def account():
	if 'loggedin' in session:
		return render_template("account.html", user_type=session['user_type'])
	return redirect(url_for('login'))

@app.route("/change_pw", methods=['GET', 'POST'])
def change_pw():
	if 'loggedin' in session:
		if request.method == 'POST' and 'new_pw' in request.form and 'confirm' in request.form:
			if request.form['new_pw'] != request.form['confirm']:
				flash('Palavra-passe não coincide', 'error')
				return render_template("change_pw.html", user_type=session['user_type'])
			else:
				password = request.form['new_pw']
				email = session['email']
				if mdb.change_password(database_name=db_name, email=email, new_password=password):
					# change sucessful
					session['default_pw'] = False
					flash('Palavra-passe alterada com sucesso', 'info')
					return render_template("change_pw.html", user_type=session['user_type'])
				else:
					# change failed
					flash('Erro ao alterar palavra-passe', 'error')
					return render_template("change_pw.html", user_type=session['user_type'])
		elif request.method == 'GET':
			return render_template("change_pw.html", user_type=session['user_type'])
		return render_template("account.html", user_type=session['user_type'])
	return redirect(url_for('login'))

# ============ MAIN ============
def main():
	app.run()

if __name__ == '__main__':
	main()