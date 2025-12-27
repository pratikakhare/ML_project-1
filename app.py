from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get CGPA from form
        cgpa = float(request.form['CGPA'])

        # Reshape for prediction
        final_features = np.array([[cgpa]])

        # Predict package
        prediction = model.predict(final_features)
        predicted_package = prediction[0]

        return render_template(
            'index.html',
            prediction_text=f'Predicted Package: {predicted_package:.2f} LPA'
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
