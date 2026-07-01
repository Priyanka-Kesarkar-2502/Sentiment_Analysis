# Sentiment Analysis on Tweets

A web application that analyzes tweets and classifies them as **Positive** or **Negative** using Machine Learning. Built with Flask for the backend and a trained model for sentiment prediction.

## 📌 Features

- Takes a tweet/text as input
- Predicts whether the sentiment is **Positive** or **Negative**
- Simple and interactive web interface built using Flask, HTML, and CSS
- Trained on real-world Twitter data

## 📊 Dataset

This project uses the **Sentiment140** dataset from Kaggle, which contains 1.6 million tweets labeled as positive or negative.

🔗 Dataset link: [Sentiment140 - Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

> Note: The dataset file is not included in this repository due to its large size (~230MB). Please download it from the link above and place it in a `dataset/` folder before running `train_model.py`.

## 🛠️ Technologies Used

- **Python**
- **Flask** – for building the web application
- **scikit-learn** – for training the machine learning model
- **Pandas / NumPy** – for data preprocessing
- **HTML/CSS** – for the frontend

## 📁 Project Structure

```
Sentiment_Analysis/
│
├── app.py                  # Flask application
├── main.py                 # Main script
├── train_model.py          # Script to train the sentiment model
├── models/                 # Saved model and vectorizer files
├── templates/               # HTML templates
├── static/                  # CSS and static files
├── .gitignore
└── README.md
```

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Priyanka-Kesarkar-2502/Sentiment_Analysis.git
   cd Sentiment_Analysis
   ```

2. **Install the required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Download `Sentiment140` dataset from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
   - Place the CSV file inside a `dataset/` folder

4. **Train the model** (optional, if you want to retrain)
   ```bash
   python train_model.py
   ```

5. **Run the Flask application**
   ```bash
   python app.py
   ```

6. Open your browser and go to:
   ```
   http://localhost:5000
   ```

## 🖼️ Screenshot

![Sentiment Analysis](Sentiment%20Analysis.png)

