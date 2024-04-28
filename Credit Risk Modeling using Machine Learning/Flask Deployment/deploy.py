from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('eps_v1.sav', 'rb'))

# Home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        v1 = request.form.get('ROCE (%)', type=float)
        v2 = request.form.get('CASA (%)', type=float)
        v3 = request.form.get('Return on Equity / Networth (%)', type=float)
        v4 = request.form.get('Non-Interest Income/Total Assets (%)', type=float)
        v5 = request.form.get('Operating Profit/Total Assets (%)', type=float)
        v6 = request.form.get('Operating Expenses/Total Assets (%)', type=float)
        v7 = request.form.get('Interest Expenses/Total Assets (%)', type=float)
        v8 = request.form.get('Face_value', type=float)

        # Make prediction
        result = model.predict([[v1,v2,v3,v4,v5,v6,v7,v8]])[0]

        # Pass input values and result back to the template
        return render_template('index.html', result=result, v1=v1, v2=v2, v3=v3, v4=v4, v5=v5, v6=v6, v7=v7, v8=v8)
    else:
        # If GET request, redirect to home page
        return redirect('/')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
