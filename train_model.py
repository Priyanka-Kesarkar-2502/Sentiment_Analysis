import pandas as pd
import re
import nltk
import joblib
import matplotlib.pyplot as plt
import os
os.makedirs("models", exist_ok=True)

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

nltk.download('stopwords')

columns = [
    "target",
    "id",
    "date",
    "flag",
    "user",
    "text"
]

df = pd.read_csv(
    "dataset/Sentiment 140.csv",
    encoding="latin-1",
    names=columns
)
df = df.sample(n=50000, random_state=42)

df = df[['target', 'text']]

print(df.head())

stop_words = set(stopwords.words('english'))

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"@\w+", "", text)

    text = re.sub(r"#", "", text)

    text = re.sub(r"[^a-zA-Z ]", "", text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)

print("\nSentiment Distribution\n")
print(df['target'].value_counts())

df['target'].value_counts().plot(kind='bar')

plt.title("Sentiment Distribution")
plt.show()

X = df['clean_text']
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer(
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

# Logistic Regression

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(
    X_train_tfidf,
    y_train
)

lr_pred = lr_model.predict(
    X_test_tfidf
)
print('LogisticRegression')
# Naive Bayes

nb_model = MultinomialNB()

nb_model.fit(
    X_train_tfidf,
    y_train
)

nb_pred = nb_model.predict(
    X_test_tfidf
)
print('Naive Bayes')
# Random Forest

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(
    X_train_tfidf,
    y_train
)

rf_pred = rf_model.predict(
    X_test_tfidf
)

print("\nLOGISTIC REGRESSION\n")

print(
    accuracy_score(
        y_test,
        lr_pred
    )
)

print(
    classification_report(
        y_test,
        lr_pred
    )
)

print(
    confusion_matrix(
        y_test,
        lr_pred
    )
)

print("\nNAIVE BAYES\n")

print(
    accuracy_score(
        y_test,
        nb_pred
    )
)

print("\nRANDOM FOREST\n")

print(
    accuracy_score(
        y_test,
        rf_pred
    )
)

joblib.dump(
    lr_model,
    "models/sentiment_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/vectorizer.pkl"
)

print("\nModel Saved Successfully")
