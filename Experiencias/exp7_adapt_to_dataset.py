import pandas as pd
from sklearn.calibration import LabelEncoder

#certifica se que a target column fica apenas uma apos encoding 
#e usa o get dummies para o resto
def analyze_dataframe(df, target_variable):

    encoding_map = build_encoding_map(df, target_variable)
    df = encode_column(df, target_variable, encoding_map)

    df = pd.get_dummies(df, drop_first=True)
    column_names = df.columns.tolist()

    for name in column_names:
        print(name)

    print(build_query(df))

#recebe um mapa e um dataframe e aplica o mapa ao dataframe
def encode_column(df, column_name, encoding_map):
    df[column_name] = df[column_name].map(encoding_map)
    return df

#funcao que pega numa coluna e constroi o mapa de encoding de acordo com o utilizador
#algo do genero mas usaria input pelo frontend
def build_encoding_map(df, column_name):
    unique_values = df[column_name].unique()
    encoding_map = {}
    for unique_value in unique_values:
        print("What is the encoding for", unique_value, "?")
        encoding = input()
        encoding_map[unique_value] = encoding
    return encoding_map

def build_query(df):
    column_names = df.columns.tolist()
    column_names = ["`" + str(name) + "` VARCHAR(255)" for name in column_names]
    column_names.append("`dataset_id` INT")
    query = "CREATE TABLE IF NOT EXISTS `data` (" + ", ".join(column_names) + ");"
    return query

filename = "C:/Users/josed/OneDrive - Ensino Lusófona/3º ano 2º semestre/Projeto II/Insight Scholaris/Experiencias/studentInfo.csv"

data = pd.read_csv(filename)

analyze_dataframe(data, "final_result")
