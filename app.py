import json
from flask import Flask, flash, render_template, request, redirect, url_for, session, send_file
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from modules import database as mdb
from modules import modeling
import pandas as pd
from dotenv import load_dotenv
import os

# ============ dotenv ============
load_dotenv() 
db_name = os.getenv('DB_NAME')

# ============ FLASK ============
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
			
		else:
			flash('Palavra-passe incorreta', 'error')
			return redirect(url_for('login'))
	elif request.method == 'POST':
		flash('Preencha todos os campos antes de submeter', 'error')
		return render_template('account/login.html')
	return render_template('account/login.html')

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
""" @app.route("/account")
@login_required
def account():
	return render_template("account.html", user_type=current_user.user_type) """

@app.route("/change_pw", methods=['GET', 'POST'])
@login_required
def change_pw():
	if request.method == 'POST' and 'new_pw' in request.form and 'pw_confirm' in request.form:
		if request.form['new_pw'] != request.form['pw_confirm']:
			flash('Palavra-passe não coincide', 'error')
			return render_template("account/change_pw.html", user_type=current_user.user_type)
		else:
			password = request.form['new_pw']
			email = current_user.email
			if mdb.change_password(database_name=db_name, email=email, new_password=password):
				# change sucessful
				current_user.default_pw = False
				flash('Palavra-passe alterada com sucesso', 'info')
				return render_template("account/change_pw.html", user_type=current_user.user_type)
			else:
				# change failed
				flash('Erro ao alterar palavra-passe', 'error')
				return render_template("account/change_pw.html", user_type=current_user.user_type)
	elif request.method == 'POST':
		flash('Preencha todos os campos antes de submeter', 'error')
		return render_template("account/change_pw.html", user_type=current_user.user_type)
	elif request.method == 'GET':
		return render_template("account/change_pw.html", user_type=current_user.user_type)
	return render_template("account/account.html", user_type=current_user.user_type)

# ============ MENU ADMIN ============
""" @app.route("/admin")
@login_required
def admin():
	return render_template("admin.html", user_type=current_user.user_type) """

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
			return render_template("admin/deactivate_acc.html", \
								user_type=current_user.user_type, \
								users_df=users_df)
		elif request.method == 'POST':
			flash('Selecione pelo menos uma conta para desativar', 'error')
			return render_template("admin/deactivate_acc.html", user_type=current_user.user_type)						
		elif request.method == 'GET':
			# get users list and render page
			users_df = mdb.list_users(database_name=db_name)

			# add checkbox column
			users_df.insert(0, '', ['<input type="checkbox" name="email" value="{}">'.format(email) for email in users_df['E-mail']])

			users_df = users_df.to_html(classes='table', index=False, escape=False)

			return render_template("admin/deactivate_acc.html", \
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
				return render_template("admin/create_user.html", user_type=current_user.user_type)

			if mdb.create_user(database_name=db_name, email=email, full_name=fullname, num_id=num_id, type=type):
				flash('Conta criada com sucesso', 'info')
				return render_template("admin/create_user.html", user_type=current_user.user_type)
			else:
				flash('Erro ao criar conta', 'error')
				return render_template("admin/create_user.html", user_type=current_user.user_type)
		elif request.method == 'POST':
			flash('Preencha todos os campos antes de submeter', 'error')
			return render_template("admin/create_user.html", user_type=current_user.user_type)					
		elif request.method == 'GET':
			return render_template("admin/create_user.html", user_type=current_user.user_type)
		
	return redirect(url_for('index'))

# ============ MENU DADOS ============
""" @app.route("/datasets")
@login_required
def datasets():
	return render_template("datasets.html", user_type=current_user.user_type) """

@app.route("/import_ds", methods=['GET', 'POST'])
@login_required
def import_ds():
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
				return render_template("datasets/import_ds.html", user_type=current_user.user_type)
		except Exception as e:
			print(e)
			flash('Erro ao carregar ficheiro', 'error')
			return render_template("datasets/import_ds.html", user_type=current_user.user_type)
		if mdb.store_dataset(db_name=db_name, df_name=ds_name, df=df, df_type=ds_type):
			flash('Ficheiro carregado com sucesso', 'info')
			return render_template("datasets/import_ds.html", user_type=current_user.user_type)
		else:
			flash('Erro ao carregar ficheiro', 'error')
			return render_template("datasets/import_ds.html", user_type=current_user.user_type)
	elif request.method == 'POST':
		flash('Preencha todos os campos antes de submeter', 'error')
		return render_template("datasets/import_ds.html", user_type=current_user.user_type)
	elif request.method == 'GET':
		return render_template("datasets/import_ds.html", user_type=current_user.user_type)

@app.route("/select_ds", methods=['GET'])
@login_required
def select_ds():
	list_df = mdb.list_datasets(database_name=db_name)
	if list_df is not None:
		list_df = list_df.to_dict(orient='records')
		return render_template("datasets/select_ds.html", user_type=current_user.user_type, list_df=list_df)
	else:
		return render_template("datasets/select_ds.html", user_type=current_user.user_type, list_df=None)
	
@app.route("/ds_menu", methods=['GET', 'POST'])
@login_required
def ds_menu():
	if request.method == 'POST' and 'id' in request.form:
		ds_id = request.form.get('id')
		if ds_id == 'None':
			flash('Selecione um dataset', 'error')
			return render_template("datasets/select_ds.html", user_type=current_user.user_type)
		ds_info = mdb.retrieve_dataset_info(database_name=db_name, df_id=ds_id)
		ds_head = mdb.retireve_head(database_name=db_name, df_id=ds_id, n_rows=5)
		ds_head = ds_head.to_html(classes='table')
		return render_template("datasets/ds_menu.html", \
						 user_type=current_user.user_type, \
							ds_info=ds_info, \
								ds_head=ds_head)
	elif request.method == 'POST' and 'ds_id' not in request.form:
		flash('Ocorreu um erro', 'error')
		return redirect(url_for('select_ds'))
	elif request.method == 'GET':
		return render_template("datasets/ds_menu.html", user_type=current_user.user_type)

@app.route("/export_ds", methods=['POST'])
@login_required
def export_ds():
	if "ds_id" in request.form:
		ds_id = request.form.get("ds_id")
		if mdb.ready_export(database_name=db_name, df_id=ds_id):
			
			ds_name = mdb.retrieve_dataset_info(database_name=db_name, df_id=ds_id)[1]
			return send_file('static/downloads/data.csv', as_attachment=True, download_name=f'{ds_name}.csv')
		
	flash('Ocorreu um erro', 'error')
	redirect(url_for('select_ds'))

""" @app.route("/all_data", methods=['GET', 'POST'])
@login_required
def all_data():
	if "ds_id" in request.form:
		ds_id = request.form.get("ds_id")
		ds_info = mdb.retrieve_dataset_info(database_name=db_name, df_id=ds_id)
		ds = mdb.retrieve_dataset(database_name=db_name, df_id=ds_id)
		ds = ds.to_html(classes='table')
		return render_template("datasets/all_data.html", user_type=current_user.user_type, ds=ds, ds_info=ds_info)
	flash('Ocorreu um erro', 'error')
	redirect(url_for('select_ds')) """

# ============ MENU PREVER ============
""" @app.route("/predict_menu")
@login_required
def predict_menu():
	return render_template("predict_menu.html", user_type=current_user.user_type) """



# ============ MENU MODELAR ============
""" @app.route("/modeling_menu")
@login_required
def modeling_menu():
	return render_template("modeling_menu.html", user_type=current_user.user_type) """

@app.route("/create_model", methods=['GET', 'POST'])
@login_required
def create_model():
	if request.method == 'POST' and 'model_name' in request.form and 'type' in request.form:
		model_name = request.form.get('model_name')
		model_type = request.form.get('type')

		if 'model_params' in session:
			session.pop('model_params', None)
		if 'model_info' in session:
			session.pop('model_info', None)
		session['model'] = {}
		session['model_params'] = {}
		session['model']['model_name'] = model_name
		session['model']['model_type'] = "SVM" if model_type == "1" else "XGBoost" if model_type == "2" else "Random Forest"
		session['model']['is_trained'] = "False"
		
		if model_type == "1": #SVM
			if 'kernel' in request.form:
				kernel = request.form.get('kernel')
				if kernel == "1": #Linear
					session['model_params']['kernel'] = "linear"
					return render_template("model/create_lsvm.html", user_type=current_user.user_type)
				elif kernel == "2": #Poly
					session['model_params']['kernel'] = "poly"
					return render_template("model/create_poly_svm.html", user_type=current_user.user_type)
				elif kernel == "3": #RBF
					session['model_params']['kernel'] = "rbf"
					return render_template("model/create_rbf_svm.html", user_type=current_user.user_type)
				else:
					flash('Erro: Kernel não suportado', 'error')
					return render_template("model/create_model.html", user_type=current_user.user_type)
			else:
				flash('Erro: Selecione um kernel', 'error')
				return render_template("model/create_model.html", user_type=current_user.user_type)
			
		elif model_type == "2": #XGBoost
			return render_template("model/create_xgb.html", user_type=current_user.user_type)
		elif model_type == "3": #Random Forest
			return render_template("model/create_rf.html", user_type=current_user.user_type)
		else:
			flash('Erro: Modelo não suportado', 'error')
			return render_template("model/create_model.html", user_type=current_user.user_type)
	elif request.method == 'POST':
		flash('Preencha todos os campos antes de submeter', 'error')
		return render_template("model/create_model.html", user_type=current_user.user_type)
	elif request.method == 'GET':
		return render_template("model/create_model.html", user_type=current_user.user_type)
	
@app.route("/create_lsvm", methods=['GET', 'POST'])
@login_required
def create_lsvm():
	model_params = session.get('model_params', None) # if there is nothing in the session, return None
	if model_params is None:
		flash('Erro: Parâmetros do modelo não encontrados', 'error')
		return render_template("model/create_model.html", user_type=current_user.user_type)

	elif request.method == 'POST':
		if 'c' in request.form:
			c = request.form.get('c')
			session['model_params']['C'] = c
		if 'shrinking' in request.form:
			shrinking = request.form.get('shrinking')
			session['model_params']['shrinking'] = shrinking
		if 'probability' in request.form:
			probability = request.form.get('probability')
			session['model_params']['probability'] = probability
		if 'random_state' in request.form:
			random_state = request.form.get('random_state')
			if random_state == 'set':
				random_state = request.form.get('random_state_value')
				session['model_params']['random_state'] = random_state
		if 'tol' in request.form:
			tol = request.form.get('tol')
			session['model_params']['tol'] = tol
		if 'cache_size' in request.form:
			cache_size = request.form.get('cache_size')
			session['model_params']['cache_size'] = cache_size
		if 'max_iter-select' in request.form:
			max_iter = request.form.get('max_iter-select')
			if max-iter == 'custom':
				max_iter = request.form.get('max_iter')
			session['model_params']['max_iter'] = max_iter
		if 'decision_function_shape' in request.form:
			decision_function_shape = request.form.get('decision_function_shape')
			session['model_params']['decision_function_shape'] = decision_function_shape
		if 'break_ties' in request.form:
			break_ties = request.form.get('break_ties')
			session['model_params']['break_ties'] = break_ties
		return render_template("teste.html", user_type=current_user.user_type, model_params=session['model_params'])
	elif request.method == 'GET':
		return render_template("model/create_lsvm.html", user_type=current_user.user_type)

@app.route("/create_rbf_svm", methods=['GET', 'POST'])
@login_required
def create_rbf_svm():
	model_params = session.get('model_params', None) # if there is nothing in the session, return None
	return render_template("model/create_rbf_svm.html", user_type=current_user.user_type)

@app.route("/create_poly_svm", methods=['GET', 'POST'])
@login_required
def create_poly_svm():
	model_params = session.get('model_params', None) # if there is nothing in the session, return None
	if model_params is None:
		flash('Erro: Parâmetros do modelo não encontrados', 'error')
		return render_template("model/create_model.html", user_type=current_user.user_type)

	elif request.method == 'POST':
		if 'c' in request.form:
			c = request.form.get('c')
			session['model_params']['C'] = c
		if 'degree' in request.form:
			degree = request.form.get('degree')
			session['model_params']['degree'] = degree
		if 'gamma' in request.form:
			gamma = request.form.get('gamma')
			if gamma == 'custom':
				gamma = request.form.get('gamma_value')
			session['model_params']['gamma'] = gamma
		if 'coef0' in request.form:
			coef0 = request.form.get('coef0')
			session['model_params']['coef0'] = coef0
		if 'shrinking' in request.form:
			shrinking = request.form.get('shrinking')
			session['model_params']['shrinking'] = shrinking
		if 'probability' in request.form:
			probability = request.form.get('probability')
			session['model_params']['probability'] = probability
		if 'random_state' in request.form:
			random_state = request.form.get('random_state')
			if random_state == 'set':
				random_state = request.form.get('random_state_value')
				session['model_params']['random_state'] = random_state
		if 'tol' in request.form:
			tol = request.form.get('tol')
			session['model_params']['tol'] = tol
		if 'cache_size' in request.form:
			cache_size = request.form.get('cache_size')
			session['model_params']['cache_size'] = cache_size
		if 'max_iter-select' in request.form:
			max_iter = request.form.get('max_iter-select')
			if max-iter == 'custom':
				max_iter = request.form.get('max_iter')
			session['model_params']['max_iter'] = max_iter
		if 'decision_function_shape' in request.form:
			decision_function_shape = request.form.get('decision_function_shape')
			session['model_params']['decision_function_shape'] = decision_function_shape
		if 'break_ties' in request.form:
			break_ties = request.form.get('break_ties')
			session['model_params']['break_ties'] = break_ties
		return render_template("teste.html", user_type=current_user.user_type, model_params=session['model_params'])
	elif request.method == 'GET':
		return render_template("model/create_poly_svm.html", user_type=current_user.user_type)

@app.route("/create_xgb", methods=['GET', 'POST'])
@login_required
def create_xgb():
	model_params = session.get('model_params', None) # if there is nothing in the session, return None
	if model_params is None:
		flash('Erro: Parâmetros do modelo não encontrados', 'error')
		return render_template("model/create_xgb.html", user_type=current_user.user_type)
	elif request.method == 'POST':
		if 'learning_rate' in request.form:
			learning_rate = float(request.form.get('learning_rate'))
			session['model_params']['learning_rate'] = learning_rate
		if 'n_estimators' in request.form:
			n_estimators = request.form.get('n_estimators')
			session['model_params']['n_estimators'] = int(n_estimators)
		if 'max_depth' in request.form:
			max_depth = request.form.get('max_depth')
			session['model_params']['max_depth'] = int(max_depth)
		if 'min_child_weight' in request.form:
			min_child_weight = request.form.get('min_child_weight')
			session['model_params']['min_child_weight'] = float(min_child_weight)
		if 'gamma' in request.form:
			gamma = request.form.get('gamma')
			session['model_params']['gamma'] = float(gamma)
		if 'subsample' in request.form:
			subsample = request.form.get('subsample')
			session['model_params']['subsample'] = float(subsample)

		model = modeling.create_xgboost_model(session['model_params'])

		model_id = mdb.store_model(database_name=db_name, model_name=session['model']['model_name'], model=model, model_type=2)
		session['model']['active_model_id'] = model_id
		session.modified = True  # This tells Flask to save the session
		# since we are just changing a value thats not at the top level of the dictionary it wont know that it has changed
		
		if  not mdb.set_active_model(database_name=db_name, model_id=model_id):
			flash('Erro ao ativar modelo', 'error')
			return render_template("model/create_xgb.html", user_type=current_user.user_type)
		else:
			list_df = mdb.list_datasets(database_name=db_name)
			if list_df is not None:
				list_df = list_df.to_dict(orient='records')
			return render_template("model/train_model.html", user_type=current_user.user_type, list_df=list_df)
	elif request.method == 'GET':
		return render_template("model/create_xgb.html", user_type=current_user.user_type)

@app.route("/create_rf", methods=['GET', 'POST'])
@login_required
def create_rf():
	model_params = session.get('model_params', None) # if there is nothing in the session, return None
	return render_template("model/create_rf.html", user_type=current_user.user_type)

@app.route("/train_model", methods=['GET', 'POST'])
@login_required
def train_model():
	list_df = mdb.list_datasets(database_name=db_name)
	if list_df is not None:
		list_df = list_df.to_dict(orient='records')

	if request.method == 'POST' and 'split' in request.form and 'id' in request.form:
		id = request.form.get('id')
		dataset = mdb.retrieve_dataset(database_name=db_name, df_id=id, decode=False)
		split = int(request.form.get('split')) / 100 # convert to percentage
		model_id = session['model']['active_model_id']
		model = mdb.retrieve_model(database_name=db_name, model_id=model_id)

		if model is not None:
			if modeling.train_model(database_name=db_name, model=model, model_id=model_id, dataset=dataset, split=split):
				flash('Modelo treinado com sucesso', 'info')
				return render_template("model/model_info.html", user_type=current_user.user_type)
			else:
				flash('Erro ao treinar modelo', 'error')
				return render_template("model/train_model.html", user_type=current_user.user_type)
		else:
			flash('Erro ao carregar modelo', 'error')
			return render_template("model/train_model.html", user_type=current_user.user_type)
	
	elif request.method == 'GET':
		return render_template("model/train_model.html", user_type=current_user.user_type, list_df=list_df)

@app.route("/model_info", methods=['GET'])
@login_required
def model_info():
	model_info = mdb.retrieve_active_model_info(database_name=db_name)
	#print(model_info.columns)
	parameters = model_info['parameters'].values[0]
	f1_score = modeling.get_f1_score(database_name=db_name, model_id=model_info['model_id'].values[0])
	return render_template("model/model_info.html", user_type=current_user.user_type, model_info=model_info, parameters=parameters, f1_score=f1_score)

@app.route("/parameters", methods=['GET'])
@login_required
def parameters():
	parameters = mdb.retrieve_active_model_info(database_name=db_name)['parameters'].values[0]
	return render_template("model/parameters.html", user_type=current_user.user_type, parameters=parameters)

@app.route("/evaluation", methods=['GET'])
@login_required
def evaluation():
	model_info = mdb.retrieve_active_model_info(database_name=db_name)
	full_eval = modeling.create_full_eval(database_name=db_name, model_id=model_info['model_id'].values[0], pt=True)
	return render_template("model/evaluation.html", user_type=current_user.user_type, full_eval=full_eval)

@app.route("/algo", methods=['GET'])
@login_required
def algo():
	return render_template("model/algo.html", user_type=current_user.user_type)


# ============ PREDICT ============
@app.route("/new_prediction", methods=['GET', 'POST'])
@login_required
def new_prediction():
	return render_template("predict/new_prediction.html", user_type=current_user.user_type)


# ============ MAIN ============
def main():
	app.run()

if __name__ == '__main__':
	main()

