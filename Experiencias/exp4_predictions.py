from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd

""" Tk().withdraw()

# Open the file picker dialog and get the path of the selected file
filename = askopenfilename()

data = pd.read_csv(filename)
print (filename) """

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

#Remove a sample (10%) of the dataframe for later predictions
sample = data.sample(frac=0.1, random_state=1)
data = data.drop(sample.index)
sample = sample.drop(columns=['final_result'])

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

#Predicting with previously removed sample data
# Save the 'id_student' column in a separate variable
id_student = sample['id_student']

# Drop the 'id_student' column from the sample data
sample_data = sample.drop(['id_student'], axis=1)

sample_data = pd.get_dummies(sample_data)

# Make predictions
predictions = clf.predict(sample_data)

# Combine 'id_student' and predictions into a DataFrame
result = pd.DataFrame({'id_student': id_student, 'final_result': predictions})

# Print the result
print("Predictions:")
print(result.head(10))
