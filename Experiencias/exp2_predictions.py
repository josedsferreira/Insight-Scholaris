import pandas as pd

filename = "C:/Users/josed/OneDrive - Ensino Lusófona/3º ano 2º semestre/Projeto II/Insight Scholaris/Experiencias/studentInfo.csv"

data = pd.read_csv(filename)

g_mapping = {'M': 0, 'F': 1}
d_mapping = {'N': 0, 'Y': 1}
fr_mapping = {'Pass': 1, 'Distinction': 1, 'Withdrawn': 0, 'Fail': 0}
age_mapping = {'0-35': 0, '35-55': 1, '55<=': 2}

data['gender'] = data['gender'].map(g_mapping)
data['disability'] = data['disability'].map(d_mapping)
data['final_result'] = data['final_result'].map(fr_mapping)
data['age_band'] = data['age_band'].map(age_mapping)

data.dropna(inplace=True)

#XGBOOST

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X = data.drop(['final_result', 'id_student'], axis=1)  # Features / colunas que vao ser usadas para prever
#tambem dropei code_module e code_presentation porque acho que nao sao relevantes para prever o resultado final

X = pd.get_dummies(X) #transforma as colunas categoricas em colunas binarias

y = data['final_result']  # Target variable / coluna que vai ser prevista

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a XGBoost Classifier
clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

print(
    f"Classification report for classifier {clf}:\n"
    f"{classification_report(y_test, y_pred)}\n"
)


new_data = pd.DataFrame({
    'code_module': ['AAA', 'BBB'],
    'code_presentation': ['2013J', '2013J'],
    'gender': [1, 0],
    'region': ['East Anglian Region', 'Scotland'],
    'highest_education': ['HE Qualification', 'A Level or Equivalent'],
    'imd_band': ['90-100%', '20-30%'],
    'age_band': [1, 3],
    'num_of_prev_attempts': [0, 1],
    'studied_credits': [60, 120],
    'disability': [1, 0]
})

# Convert categorical variables into binary
new_data = pd.get_dummies(new_data)

# Use the trained model to make predictions
new_predictions = clf.predict(new_data)

# Now new_predictions contains the predicted values
print("New predictions")
print(new_predictions)