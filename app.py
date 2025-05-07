from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open('diamond_model.pkl', 'rb') as f:
    model = pickle.load(f)
@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template("model.html")
    elif request.method == 'POST':
        carat = float(request.form['carat'])
        cut = int(request.form['cut'])
        color = int(request.form['color'])
        clarity = int(request.form['clarity'])

        features = np.array([[carat, cut, color, clarity]])
        prediction = round(model.predict(features)[0],2)
        return f"<h2>Predicted Price: ${prediction:.2f}</h2><br><a href='/'>Try another</a>"
    return render_template ('model.html')
if __name__ == '__main__':
    app.run(debug=True)
