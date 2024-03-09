from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import cleaning as cl

filename = "C:/Users/josed/OneDrive - Ensino Lusófona/3º ano 2º semestre/Projeto II/Insight Scholaris/Experiencias/studentInfo.csv"

df = pd.read_csv(filename)

df = cl.clean_data(df)

engine = create_engine('postgresql://postgres:postgrespw@localhost:5432/exploration1')

