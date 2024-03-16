
CREATE TABLE dataframes (
    df_id SERIAL PRIMARY KEY,
    df_name VARCHAR(255),
    df_type integer, /* 1- de treino, 2- por prever, 3- já previsto */
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

create table dataframe_changes (
    change_id SERIAL PRIMARY KEY,
    df_id INTEGER REFERENCES dataframes(df_id),
    change_description VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

/* O dataframe será guardado de modo encoded para ser mais eficiente */
/*
tipo 1 - de treino, usa todas as colunas
tipo 2 - por prever, não usa coluna final_result
tipo 3 - já previsto, usa todas as colunas (se calhar este tipo não é necessário)
*/
create TABLE data (
    data_id SERIAL PRIMARY KEY,
    dataframe_id INTEGER REFERENCES dataframes(dataframe_id),
    code_module INTEGER,
    code_presentation INTEGER,
    id_student INTEGER,
    gender INTEGER,
    region INTEGER,
    highest_education INTEGER,
    imd_band INTEGER,
    age_band INTEGER,
    num_of_prev_attempts INTEGER,
    studied_credits INTEGER,
    disability INTEGER,
    final_result INTEGER
);

/* O modelo será guardado no computador e na base de dados apenas se guarda o 
nome do ficheiro, o resto do file path fica no .env */
CREATE Table models (
    model_id SERIAL PRIMARY KEY,
    model_name VARCHAR(255),
    model_type integer, /* 1-SVM, 2-XGB, 3-RF*/
    model_file_name VARCHAR(255),
    parameters JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

/* basta o fp, fn, tp e tn para obter todas as metricas de avaliação */
create table evaluations (
    evaluation_id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES models(model_id),
    fp INTEGER,
    fn INTEGER,
    tp INTEGER,
    tn INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

/*os parametros afinal vao ser guardados na tabela do modelo*/
/* create table parameters (
    parameters_id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES models(model_id),
    parameters JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
); */

/* Contas */
create table users (
    user_id SERIAL PRIMARY KEY,
    full_name VARCHAR(255),
    email varchar(255) UNIQUE NOT NULL,
    num_id integer,
    type integer, /* 1-Administrador, 2-Docente, 3-CientistaDados */
    is_active BOOLEAN DEFAULT true,
    password VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE administrators (
    admin_id SERIAL PRIMARY KEY REFERENCES users(user_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE teachers (
    teacher_id SERIAL PRIMARY KEY REFERENCES users(user_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE scientists (
    scientist_id SERIAL PRIMARY KEY REFERENCES users(user_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);