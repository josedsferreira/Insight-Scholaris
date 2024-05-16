# Insight Scholaris - Academic Success Prediction Web App

A web app designed to predict students' academic success and enable educators to intervene to improve outcomes. 
It allows for the creation of predictive models and their utilization to predict whether students will pass the course.
It uses a Flask server and a Postgres database.

To run:

install the dependencies and create a Postgres database using the init.sql file.

Create a .env file with the following format:
USER_POSTGRES = <Postgres username>
PASSWORD_POSTGRES = <Postgres password>
COLUMN_NAMES="code_module,code_presentation,id_student,gender,region,highest_education,imd_band,age_band,num_of_prev_attempts,studied_credits,disability,final_result"
DUMMIES_COL_NAMES="gender_1,gender_2,gender_3,gender_0,disability_1,disability_2,disability_0,age_band_1,age_band_2,age_band_3,age_band_0,highest_education_1,highest_education_2,highest_education_3,highest_education_4,highest_education_5,highest_education_0,imd_band_1,imd_band_2,imd_band_3,imd_band_4,imd_band_5,imd_band_6,imd_band_7,imd_band_8,imd_band_9,imd_band_10,imd_band_0,code_presentation_1,code_presentation_2,code_presentation_3,code_presentation_4,code_presentation_0,code_module_1,code_module_2,code_module_3,code_module_4,code_module_5,code_module_6,code_module_7,code_module_0,region_1,region_2,region_3,region_4,region_15,region_6,region_7,region_8,region_9,region_10,region_11,region_12,region_13,region_14,region_0"
CATEGORICAL_COLUMN_NAMES="code_module,code_presentation,gender,region,highest_education,imd_band,age_band,disability"
SECRET_KEY = <a secret key>
DB_NAME = <database name>
MODEL_FOLDER_PATH = <folder where model files should be saved>

Create an admin user with the create_user() function.

Run the command python app.py.


# (Portuguese Version) Insight Scholaris - Web App de previsão de sucesso académico

Web App que tem o objetivo de prever o sucesso académico dos estudantes e permitir 
que os docentes intervenham para melhorar os resultados.
Permite criar modelos preditivos e usa-los para fazer se os alunos irão passar á cadeira
Usa um servidor flask e uma base de dados Postgres

Para correr instale as dependencias e crie uma base de dados Postgres a partir do ficheiro init.sql

Crie um ficheiro .env com o seguinte formato:
USER_POSTGRES = <nome de utilizador postgres>
PASSWORD_POSTGRES = <password postgres>
COLUMN_NAMES="code_module,code_presentation,id_student,gender,region,highest_education,imd_band,age_band,num_of_prev_attempts,studied_credits,disability,final_result"
DUMMIES_COL_NAMES="gender_1,gender_2,gender_3,gender_0,disability_1,disability_2,disability_0,age_band_1,age_band_2,age_band_3,age_band_0,highest_education_1,highest_education_2,highest_education_3,highest_education_4,highest_education_5,highest_education_0,imd_band_1,imd_band_2,imd_band_3,imd_band_4,imd_band_5,imd_band_6,imd_band_7,imd_band_8,imd_band_9,imd_band_10,imd_band_0,code_presentation_1,code_presentation_2,code_presentation_3,code_presentation_4,code_presentation_0,code_module_1,code_module_2,code_module_3,code_module_4,code_module_5,code_module_6,code_module_7,code_module_0,region_1,region_2,region_3,region_4,region_15,region_6,region_7,region_8,region_9,region_10,region_11,region_12,region_13,region_14,region_0"
CATEGORICAL_COLUMN_NAMES="code_module,code_presentation,gender,region,highest_education,imd_band,age_band,disability"
SECRET_KEY = <uma chave secreta>
DB_NAME = <nome da base de dados>
MODEL_FOLDER_PATH = <pasta onde se pretende guardar os ficheiros dos modelos>

criar um utilizador adminsitrador com a função create_user()

correr o comando python app.py
