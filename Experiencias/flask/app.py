from flask import Flask, render_template, redirect, url_for
from forms import MyForm

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

@app.route('/', methods=['GET', 'POST'])
def home():
    form = MyForm()
    if form.validate_on_submit():
        return redirect(url_for('thank_you', name=form.name.data))
    return render_template('home.html', form=form)

@app.route('/thank_you/<name>')
def thank_you(name):
    return render_template('thank_you.html', name=name)

if __name__ == '__main__':
    app.run(debug=True)
