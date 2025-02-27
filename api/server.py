from flask import Flask, request, jsonify
from flask_cors import CORS
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
CORS(app)

def retry_click(driver, selector, max_attempts=3):
    """Retries clicking an element if it fails."""
    for attempt in range(max_attempts):
        try:
            element = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
            driver.execute_script("arguments[0].click();", element)
            return True
        except:
            print(f"Retry {attempt + 1} failed")
            time.sleep(2)
    return False

def setup_driver():
    """Sets up the Selenium WebDriver with necessary options."""
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1080')
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

def scrape_reviews(url):
    """Scrapes reviews from Daraz and saves them in a CSV file."""
    driver = setup_driver()
    wait = WebDriverWait(driver, 20)
    reviews_list = []
    
    try:
        # Load page and wait
        print("Loading URL...")
        driver.get(url)
        time.sleep(5)
        
        # Scroll steps to trigger lazy loading
        for scroll in range(0, 2000, 200):
            driver.execute_script(f"window.scrollTo(0, {scroll})")
            time.sleep(1)
        
        # Wait for and click the reviews section
        print("Finding reviews section...")
        review_tab = wait.until(EC.element_to_be_clickable(
            (By.CSS_SELECTOR, '[data-spm-anchor-id*="review"]')))
        driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth'});", review_tab)
        time.sleep(2)
        
        if not retry_click(driver, '[data-spm-anchor-id*="review"]'):
            print("Failed to click reviews tab")
            return 0
        
        page = 1
        while True:
            print(f"Processing page {page}")
            
            # Wait for reviews container
            reviews = wait.until(EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, '.mod-reviews .item')))
            
            for review in reviews:
                try:
                    text = review.find_element(By.CSS_SELECTOR, '.content').text
                    date = review.find_element(By.CSS_SELECTOR, '.top').text
                    author = review.find_element(By.CSS_SELECTOR, '.middle').text
                    
                    if text.strip():
                        reviews_list.append({
                            'reviewText': text.strip(),
                            'reviewDate': date.strip(),
                            'authorName': author.strip()
                        })
                        print(f"Found review #{len(reviews_list)}")
                except StaleElementReferenceException:
                    continue
            
            # Save progress
            if reviews_list:
                df = pd.DataFrame(reviews_list)
                df.to_csv('reviews.csv', index=False)
            
            # Try clicking the next page
            try:
                
                next_button = driver.find_element(By.XPATH, '//button[contains(@class, "next-pagination-item next")]')
                next_button.click()
                if 'ant-pagination-disabled' in next_button.get_attribute("class"):
                    print("No more pages")
                    break
                driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth'});", next_button)
                time.sleep(2)
                driver.execute_script("arguments[0].click();", next_button)
                page += 1
                time.sleep(3)
            except Exception as e:
                print(f"Pagination error: {str(e)}")
                print("No more pages")
                break
                
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        driver.quit()
    
    return len(reviews_list)

def analyze_reviews():
    """Analyzes sentiment and generates a report from the scraped reviews."""
    df = pd.read_csv('reviews.csv')
    analyzer = SentimentIntensityAnalyzer()
    # Sentiment Analysis
    df['sentiment_scores'] = df['reviewText'].apply(analyzer.polarity_scores)
    df['sentiment'] = df['sentiment_scores'].apply(
        lambda x: 'positive' if x['compound'] > 0 else 'negative' if x['compound'] < 0 else 'neutral')
    # Generate sentiment plot
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='sentiment')
    plt.title('Sentiment Distribution')
    if not os.path.exists('static'):
        os.makedirs('static')
    plt.savefig('static/sentiment.png')
    plt.close()
    sentiment_distribution = df['sentiment'].value_counts(normalize=True).to_dict()
    # Ensure keys are lowercase for frontend compatibility
    sentiment_distribution = {k.lower(): v * 100 for k, v in sentiment_distribution.items()}
    return {
        'confidence_score': float((df['sentiment_scores'].apply(lambda x: x['compound']).mean() + 1) / 2 * 10),
        'total_reviews': len(df),
        'sentiment_distribution': sentiment_distribution,
        'sentiment_plot': 'static/sentiment.png'
    }

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint to analyze product reviews."""
    try:
        url = request.json.get('url')
        if not url:
            return jsonify({"error": "URL is required"}), 400
            
        num_reviews = scrape_reviews(url)
        if num_reviews == 0:
            return jsonify({"error": "No reviews found"}), 404
            
        results = analyze_reviews()
        
        return jsonify({
            'success': True,
            'data': results
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/previous-analyses', methods=['GET'])
def get_previous_analyses():
    if not os.path.exists('analysis_results.json'):
        return jsonify([])
    with open('analysis_results.json', 'r') as f:
        data = json.load(f)
    return jsonify(data)
if __name__ == '__main__':
    app.run(port=8000, debug=True)
