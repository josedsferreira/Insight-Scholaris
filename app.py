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
		if request.method == 'POST' and 'new_pw' in request.form and 'pw_confirm' in request.form:
			if request.form['new_pw'] != request.form['pw_confirm']:
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

# ============ MENU ADMIN ============
@app.route("/admin")
def admin():
	if 'loggedin' in session:
		return render_template("admin.html", user_type=session['user_type'])
	return redirect(url_for('login'))

@app.route("/deactivate_acc", methods=['GET', 'POST'])
def deactivate_acc():
	if 'loggedin' in session:
		if session['user_type'] == 1:
			if request.method == 'POST' and "email" in request.form:
				email_list = request.form.getlist('email')
				for email in email_list:
					if mdb.deactivate_user(database_name=db_name, email=email):
						continue
					else:
						flash('Erro ao desativar conta', 'error')
				# get users list and render page
				users_df = mdb.list_users(database_name=db_name)

				# add checkbox column
				users_df.insert(0, '', ['<input type="checkbox" name="email" value="{}">'.format(email) for email in users_df['E-mail']])

				users_df = users_df.to_html(classes='table', index=False, escape=False)
				flash('Conta(s) desativada(s) com sucesso', 'info')
				return render_template("deactivate_acc.html", \
								user_type=session['user_type'], \
								users_df=users_df)
								
			elif request.method == 'GET':
				# get users list and render page
				users_df = mdb.list_users(database_name=db_name)

				# add checkbox column
				users_df.insert(0, '', ['<input type="checkbox" name="email" value="{}">'.format(email) for email in users_df['E-mail']])

				users_df = users_df.to_html(classes='table', index=False, escape=False)

				return render_template("deactivate_acc.html", \
								user_type=session['user_type'], \
								users_df=users_df)
		
		return redirect(url_for('index'))
	return redirect(url_for('login'))

@app.route("/create_user", methods=['GET', 'POST'])
def create_user():
	if 'loggedin' in session:
		if session['user_type'] == 1:
			if request.method == 'POST' and "email" in request.form and "fullname" in request.form and "num_id" and "type" in request.form:
				email = request.form.get('email')
				fullname = request.form.get('fullname')
				type = int(request.form.get('type'))
				try:
					num_id = int(request.form.get('num_id'))
				except ValueError:
					flash('Número ID inválido', 'error')
					return render_template("create_user.html", user_type=session['user_type'])

				if mdb.create_user(database_name=db_name, email=email, full_name=fullname, num_id=num_id, type=type):
					flash('Conta criada com sucesso', 'info')
					return render_template("create_user.html", user_type=session['user_type'])
				else:
					flash('Erro ao criar conta', 'error')
					return render_template("create_user.html", user_type=session['user_type'])
								
			elif request.method == 'GET':
				return render_template("create_user.html", user_type=session['user_type'])
		
		return redirect(url_for('index'))
	return redirect(url_for('login'))

# ============ MAIN ============
def main():
	app.run()

if __name__ == '__main__':
	main()

