CREATE TABLE dataFrames (
    df_id SERIAL PRIMARY KEY,
    df_name VARCHAR(255),
    df_type integer, /* 1- de treino, 2- por prever, 3- já previsto */
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
    dataframe_id INTEGER REFERENCES DataFrames(dataframe_id),
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
    final_result INTEGER,
);

/* O modelo será guardado no computador e na base de dados apenas se guarda o file path */
CREATE Table models (
    model_id SERIAL PRIMARY KEY,
    model_name VARCHAR(255),
    model_type VARCHAR(255),
    model_file_name VARCHAR(255),
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

/* o __str__ do modelo contem os parametros todos usados */
create table parameters (
    parameter_id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES models(model_id),
    parameter_list VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

