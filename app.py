from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField
import torch
import torch.nn as nn
import pickle
import numpy as np
def return_prediction(model,scaler,sample_json):
    
    feat1 = sample_json['feat1']
    feat2 = sample_json['feat2']
  
    # Predict on a new data point (remember to scale the features!)
    new_gem = np.array([[feat1, feat2]])
    new_gem_scaled = scaler.transform(new_gem)
    new_gem_tensor = torch.tensor(new_gem_scaled, dtype=torch.float32)

    # Use the loaded model to predict on the new sample
    with torch.no_grad():
        predict = model(new_gem_tensor).item()
    
    return predict

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

# flower_model = load_model("my_model.h5")

# To load the model later, define the same architecture and load the state dictionary
flower_model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)
flower_model.load_state_dict(torch.load('model.pth', weights_only=True))
flower_model.eval()

scaler = pickle.load(open('scaler.sav', 'rb'))

class FlowerForm(FlaskForm):
    feat1 = StringField('Feature 1')
    feat2 = StringField('Feature 2')

    submit = SubmitField('Predict')

@app.route('/', methods=['GET', 'POST'])
def index():

    form = FlowerForm()
    if form.validate_on_submit():

        session['feat1'] = form.feat1.data
        session['feat2'] = form.feat2.data

        return redirect(url_for("prediction"))

    return render_template('home.html', form=form)


@app.route('/prediction')
def prediction():
    content = {}

    content['feat1'] = float(session['feat1'])
    content['feat2'] = float(session['feat2'])

    results = return_prediction(model=flower_model,scaler=scaler,sample_json=content)
    gerundeter_wert = round(results * 20) / 20
    res = round(gerundeter_wert, 2)
    return render_template('prediction.html',results=res)


if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0', port=5002) 
