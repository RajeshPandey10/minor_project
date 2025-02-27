from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from deep_translator import GoogleTranslator
from langdetect import detect

# for amazon reviews

app = Flask(__name__)
CORS(app)

# Your ScraperAPI key
SCRAPERAPI_KEY = "5ea6d341200743e19427a02d210e496f"

def scrape_amazon_reviews(url):
    scraperapi_url = f"http://api.scraperapi.com/?api_key={SCRAPERAPI_KEY}&url={url}"
    response = requests.get(scraperapi_url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    review_elements = soup.find_all("span", {"data-hook": "review-body"})
    reviews = [review.get_text(strip=True) for review in review_elements]
    return reviews

# Load the Logistic Regression model and vectorizer
with open('model_LogisticRegression.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Function to analyze reviews using the Logistic Regression model
def analyze_reviews():
    # Load the scraped reviews
    df = pd.read_csv('reviews.csv')

    # Handle missing values in the reviewText column
    df['reviewText'] = df['reviewText'].fillna('')  # Replace NaN with an empty string

    # Function to translate text to English
    def translate_to_english(text):
        try:
            lang = detect(text)  # Detect language
            if lang != "en":  # If not English, translate
                return GoogleTranslator(source='auto', target='en').translate(text)
            return text  # If already English, return as is
        except:
            return text  # If detection fails, return original

    # Apply translation to reviewText column
    df["translated_review"] = df["reviewText"].astype(str).apply(translate_to_english)

    # Save the translated dataset
    df.to_csv("translated_reviews.csv", index=False)

    print("Translation completed and saved!")

    # Transform reviews using the saved vectorizer
    X_test = vectorizer.transform(df['translated_review'])

    # Make predictions with the RandomForest model
    predictions = model.predict(X_test)

    # Manually map numeric predictions to sentiment labels
    sentiment_mapping = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }

    # Apply mapping to predictions
    df['sentiment'] = [sentiment_mapping[pred] for pred in predictions]

    sentiment_distribution = df['sentiment'].value_counts(normalize=True).to_dict()
    # Ensure keys are lowercase for frontend compatibility
    sentiment_distribution = {k.lower(): v * 100 for k, v in sentiment_distribution.items()}

    # Mapping sentiment labels to numerical scores
    sentiment_scores = {
        "Negative": -1,
        "Neutral": 0,
        "Positive": 1
    }

    # Function to calculate compound score based on sentiment labels
    def get_compound_score(sentiment):
        return sentiment_scores.get(sentiment, 0)  # Default to 0 if sentiment is missing

    # Apply the function to calculate compound scores for each review
    df['compound'] = df['sentiment'].apply(get_compound_score)

    # Save the results to a new CSV file
    df[['reviewText', 'sentiment', 'compound']].to_csv('review_with_compound_scores.csv', index=False)

    # Calculate the overall compound score
    compound_score = (df['compound'].mean()+1)/2 * 10  # Scale compound score to 0-10

    print("CONFIDENCE SCORE".center(50, '-'))
    # Print the overall compound score
    print(f"Overall Compound Score for the product is: {compound_score:.2f} out of 10")
    # Return the results
    return {
        'confidence_score': compound_score,
        'total_reviews': len(df),
        'sentiment_distribution': sentiment_distribution,
        'sentiment_plot': 'static/sentiment.png'
    }

@app.route("/", methods=["POST"])
def scrape_csv():
    data = request.get_json()
    amazon_url = data.get("url") if data else None
    if not amazon_url:
        return jsonify({"error": "Please provide an Amazon product URL in JSON format."}), 400
    
    reviews = scrape_amazon_reviews(amazon_url)
    df = pd.DataFrame(reviews, columns=["reviewText"])
    csv_filename = "reviews.csv"
    df.to_csv(csv_filename, index=False, encoding="utf-8")
    
    print("Reviews saved to reviews.csv")

    # Analyze the reviews using the Logistic Regression model
    results = analyze_reviews()
    
    # Visualization: Sentiment Distribution
    sentiment_counts = results['sentiment_distribution']
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(sentiment_counts.keys()), y=list(sentiment_counts.values()), palette='viridis')
    plt.title('Sentiment Distribution', fontsize=16)
    plt.xlabel('Sentiment', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Ensure the "static" directory exists before saving the file
    if not os.path.exists('static'):
        os.makedirs('static')

    plt.savefig('static/sentiment.png')  # Save the plot as an image
    plt.close()
    print("Bar Plot saved to static/sentiment.png")

    return jsonify({
            'success': True,
            'message': f'Successfully scraped reviews.',
            'url': amazon_url,
            'data': results
        }), 200

if __name__ == "__main__":
    app.run(port=8000, debug=True)
