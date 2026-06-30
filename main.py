import joblib

model = joblib.load(
    "models/sentiment_model.pkl"
)

vectorizer = joblib.load(
    "models/vectorizer.pkl"
)

tweet = input(
    "Enter Tweet: "
)

tweet_vector = vectorizer.transform(
    [tweet]
)

prediction = model.predict(
    tweet_vector
)

if prediction[0] == 0:

    print("\nNegative Sentiment")

else:

    print("\nPositive Sentiment")