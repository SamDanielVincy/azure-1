from flask import Flask, render_template, request
import joblib
import numpy as np

# Load trained model
model = joblib.load("model.joblib")
#qwee
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Get form data
            sepal_length = float(request.form["sepal_length"])
            sepal_width = float(request.form["sepal_width"])
            petal_length = float(request.form["petal_length"])
            petal_width = float(request.form["petal_width"])

            # Predict
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            pred_class = model.predict(features)[0]
            classes = ["Setosa", "Versicolor", "Virginica"]
            prediction = classes[pred_class]

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
