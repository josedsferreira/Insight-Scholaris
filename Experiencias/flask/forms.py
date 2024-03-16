from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class MyForm(FlaskForm):
    name = StringField('Nome', validators=[DataRequired()])
    submit = SubmitField('Enviar')
