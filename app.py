from flask import Flask, flash, render_template, request, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from modules import database as mdb
import pandas as pd
from dotenv import load_dotenv
import os

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
app.debug = False # se true mudanças nos ficheiros automtaticamente recarregam servidor

# =========== login manager ==========
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = "Por favor faça login para ver a pagina."
login_manager.login_message_category = "error"

class User(UserMixin):
	def __init__(self, email, user_type, default_pw):
		self.email = email
		self.user_type = user_type
		self.default_pw = default_pw

	def get_id(self):
		return self.email
	
@login_manager.user_loader
def load_user(email):
	user_type, default_pw = mdb.user_info(database_name=db_name, email=email)
	return User(email, user_type, default_pw)

# ============ dotenv ============
load_dotenv()
db_name = os.getenv('DB_NAME')

# ============ ROUTES ============
@app.route('/')
@login_required
def index():
	return render_template('index.html', user_type=current_user.user_type, default_pw=current_user.default_pw)


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
			user = load_user(email)
			login_user(user)
			return redirect(url_for('index'))
			"""
			#metodo anterior
			session['loggedin'] = True
			session['email'] = email
			session['user_type'], session['default_pw'] = mdb.user_info(database_name=db_name, email=email)
			"""
			
		else:
			flash('Palavra-passe incorreta', 'error')
			return redirect(url_for('login'))
	elif request.method == 'POST':
		flash('Preencha todos os campos antes de submeter', 'error')
		return render_template('login.html')
	return render_template('login.html')

@app.route('/logout')
def logout():
	logout_user()
	""" 
	#metodo anterior
	session.pop('loggedin', None)
	session.pop('email', None)
	session.pop('type', None)
	session.pop('default_pw', None) 
	"""
	return redirect(url_for('login'))

# ============ MENU CONTA ============
@app.route("/account")
@login_required
def account():
	return render_template("account.html", user_type=current_user.user_type)

@app.route("/change_pw", methods=['GET', 'POST'])
@login_required
def change_pw():
	if request.method == 'POST' and 'new_pw' in request.form and 'pw_confirm' in request.form:
		if request.form['new_pw'] != request.form['pw_confirm']:
			flash('Palavra-passe não coincide', 'error')
			return render_template("change_pw.html", user_type=current_user.user_type)
		else:
			password = request.form['new_pw']
			email = session['email']
			if mdb.change_password(database_name=db_name, email=email, new_password=password):
				# change sucessful
				current_user.default_pw = False
				flash('Palavra-passe alterada com sucesso', 'info')
				return render_template("change_pw.html", user_type=current_user.user_type)
			else:
				# change failed
				flash('Erro ao alterar palavra-passe', 'error')
				return render_template("change_pw.html", user_type=current_user.user_type)
	elif request.method == 'POST':
		flash('Preencha todos os campos antes de submeter', 'error')
		return render_template("change_pw.html", user_type=current_user.user_type)
	elif request.method == 'GET':
		return render_template("change_pw.html", user_type=current_user.user_type)
	return render_template("account.html", user_type=current_user.user_type)

# ============ MENU ADMIN ============
@app.route("/admin")
@login_required
def admin():
	return render_template("admin.html", user_type=current_user.user_type)

@app.route("/deactivate_acc", methods=['GET', 'POST'])
@login_required
def deactivate_acc():
	if current_user.user_type == 1:
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
								user_type=current_user.user_type, \
								users_df=users_df)
		elif request.method == 'POST':
			flash('Selecione pelo menos uma conta para desativar', 'error')
			return render_template("deactivate_acc.html", user_type=current_user.user_type)						
		elif request.method == 'GET':
			# get users list and render page
			users_df = mdb.list_users(database_name=db_name)

			# add checkbox column
			users_df.insert(0, '', ['<input type="checkbox" name="email" value="{}">'.format(email) for email in users_df['E-mail']])

			users_df = users_df.to_html(classes='table', index=False, escape=False)

			return render_template("deactivate_acc.html", \
								user_type=current_user.user_type, \
								users_df=users_df)
	return redirect(url_for('index'))


@app.route("/create_user", methods=['GET', 'POST'])
@login_required
def create_user():
	if current_user.user_type == 1:
		if request.method == 'POST' and "email" in request.form and "fullname" in request.form and "num_id" and "type" in request.form:
			email = request.form.get('email')
			fullname = request.form.get('fullname')
			type = int(request.form.get('type'))
			try:
				num_id = int(request.form.get('num_id'))
			except ValueError:
				flash('Número ID inválido', 'error')
				return render_template("create_user.html", user_type=current_user.user_type)

			if mdb.create_user(database_name=db_name, email=email, full_name=fullname, num_id=num_id, type=type):
				flash('Conta criada com sucesso', 'info')
				return render_template("create_user.html", user_type=current_user.user_type)
			else:
				flash('Erro ao criar conta', 'error')
				return render_template("create_user.html", user_type=current_user.user_type)
		elif request.method == 'POST':
			flash('Preencha todos os campos antes de submeter', 'error')
			return render_template("create_user.html", user_type=current_user.user_type)					
		elif request.method == 'GET':
			return render_template("create_user.html", user_type=current_user.user_type)
		
		return redirect(url_for('index'))

# ============ MENU DADOS ============
@app.route("/datasets")
@login_required
def datasets():
	return render_template("datasets.html", user_type=current_user.user_type)

@app.route("/import_ds", methods=['GET', 'POST'])
@login_required
def import_ds():

	""" #DEBUGGING
	if request.method == 'POST':
		ds_name = request.form.get('ds_name')
		ds_file = request.files['ds_file']
		ds_type = request.form.get('ds_type')
		print(ds_type)
		print(ds_file.filename)
		print(ds_name) """
	
	if request.method == 'POST' and 'ds_file' in request.files and 'ds_name' in request.form and 'ds_type' in request.form:
		ds_name = request.form.get('ds_name')
		ds_file = request.files['ds_file']
		ds_type = request.form.get('ds_type')
		try:
			if ds_file.filename.endswith('.csv'):
				df = pd.read_csv(ds_file)
			elif ds_file.filename.endswith('.xlsx'):
				df = pd.read_excel(ds_file)
			else:
				flash('Formato de ficheiro não suportado', 'error')
				return render_template("import_ds.html", user_type=current_user.user_type)
		except Exception as e:
			print(e)
			flash('Erro ao carregar ficheiro', 'error')
			return render_template("import_ds.html", user_type=current_user.user_type)
		if mdb.store_dataset(db_name=db_name, df_name=ds_name, df=df, df_type=ds_type):
			flash('Ficheiro carregado com sucesso', 'info')
			return render_template("import_ds.html", user_type=current_user.user_type)
		else:
			flash('Erro ao carregar ficheiro', 'error')
			return render_template("import_ds.html", user_type=current_user.user_type)
	elif request.method == 'POST':
		flash('Preencha todos os campos antes de submeter', 'error')
		return render_template("import_ds.html", user_type=current_user.user_type)
	elif request.method == 'GET':
		return render_template("import_ds.html", user_type=current_user.user_type)
	
# ============ MENU PREVER ============
@app.route("/predict_menu")
@login_required
def predict_menu():
	return render_template("predict_menu.html", user_type=current_user.user_type)

# ============ MENU MODELAR ============
@app.route("/modeling_menu")
@login_required
def modeling_menu():
	return render_template("modeling_menu.html", user_type=current_user.user_type)

# ============ MAIN ============
def main():
	app.run()

if __name__ == '__main__':
	main()

