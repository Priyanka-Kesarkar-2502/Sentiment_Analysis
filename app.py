from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        tweet = request.form["tweet"]

        tweet_vector = vectorizer.transform([tweet])

        result = model.predict(tweet_vector)

        if result[0] == 0:
            prediction = "Negative Sentiment"
        else:
            prediction = "Positive Sentiment"

    return render_template(
        "index.html",
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)