// Remove the static API_URL constant and add a helper function
function getApiUrl(url) {
    if (url.includes('amazon')) {
        return 'https://api-for-minor-project.onrender.com';
    } else if (url.includes('daraz')) {
        return 'http://127.0.0.1:8000';
    } else {
        // default to onrender.com if neither match
        return 'https://api-for-minor-project.onrender.com';
    }
}

// Enhanced scraping with language support
async function scrapeReviews() {
  return new Promise((resolve, reject) => {
    chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
      try {
        const results = await chrome.scripting.executeScript({
          target: { tabId: tabs[0].id },
          func: () => {
            const reviews = [];
            document.querySelectorAll('.mod-reviews').forEach(review => {
              const content = review.querySelector('.content');
              const top = review.querySelector('.top');
              const middle = review.querySelector('.middle');
              if (content) {
                reviews.push({
                  reviewHTML: content.innerHTML.trim(),
                  reviewDate: top?.textContent.trim() || '',
                  authorName: middle?.textContent.trim() || ''
                });
              }
            });
            return reviews;
          }
        });
        resolve(results[0].result);
      } catch (error) {
        reject(new Error('Failed to scrape reviews. Try scrolling down first!'));
      }
    });
  });
}

// Main analysis flow
async function runAnalysis(elements) {
  try {
    elements.analyzeBtn.disabled = true;
    elements.loadingState.classList.remove('hidden');
    elements.results?.classList.add('hidden');
    document.getElementById('loadingSpinner').classList.remove('hidden');
    document.getElementById('sentimentPlot').classList.add('hidden');
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (!tab.url.includes('amazon') && !tab.url.includes('daraz')) {
      throw new Error('Please navigate to a product page');
    }
    
    elements.loadingText.textContent = "Analyzing reviews...";
    elements.progressBar.style.width = "50%";
    
    const apiUrl = getApiUrl(tab.url);
    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: tab.url })
    });
    
    const text = await response.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch (e) {
      console.error('JSON Parse Error:', text);
      throw new Error('Invalid response from server');
    }
    if (!data.success || !data.data) {
      throw new Error(data.error || 'Analysis failed');
    }
    const results = data.data;
    // Validate and update results
    elements.confidenceScore.textContent =
      (parseFloat(results.confidence_score) || 0).toFixed(1);
    elements.reviewCount.textContent =
      parseInt(results.total_reviews) || 0;
    if (results.sentiment_distribution) {
      updateSentimentDisplay(elements, {
        positive: parseFloat(results.sentiment_distribution.positive) || 0,
        negative: parseFloat(results.sentiment_distribution.negative) || 0,
        neutral: parseFloat(results.sentiment_distribution.neutral) || 0
      });
    }
    // Prevent caching and use full URL for the sentiment image based on page URL
    const timestamp = new Date().getTime();
    let sentimentPlotUrl;
    if (tab.url.includes('daraz')) {
        sentimentPlotUrl = `http://localhost:8000/${results.sentiment_plot}?t=${timestamp}`;
    } else {
        sentimentPlotUrl = `https://api-for-minor-project.onrender.com/static/sentiment.png?t=${timestamp}`;
    }

    document.getElementById('sentimentPlot').src = sentimentPlotUrl;
    document.getElementById('sentimentPlot').classList.remove("hidden");
    document.getElementById('loadingSpinner').classList.add("hidden");

    // Update recommendation based on confidence score
    updateRecommendation(elements, parseFloat(results.confidence_score) || 0);

    elements.loadingState.classList.add('hidden');
    elements.results.classList.remove('hidden');
  } catch (error) {
    console.error('Analysis error:', error);
    elements.loadingText.textContent = `Error: ${error.message}`;
    elements.loadingState.classList.add('error');
  } finally {
    elements.analyzeBtn.disabled = false;
  }
}

function updateRecommendation(elements, confidenceScore) {
  const recommendationContainer = document.getElementById('recommendations');
  recommendationContainer.innerHTML = ''; // Clear previous recommendations

  let recommendationText = '';
  let recommendationClass = '';

  if (confidenceScore >= 7) {
    recommendationText = 'Excellent! You can go with this product.';
    recommendationClass = 'bg-green-100 text-green-800';
  } else if (confidenceScore >= 5) {
    recommendationText = ' Average product.Could be better.';
    recommendationClass = 'bg-yellow-100 text-yellow-800';
  } else {
    recommendationText = 'Not recommended. Consider other options.';
    recommendationClass = 'bg-red-100 text-red-800';
  }

  const recommendationDiv = document.createElement('div');
  recommendationDiv.className = `p-4 rounded ${recommendationClass}`;
  recommendationDiv.textContent = recommendationText;

  recommendationContainer.appendChild(recommendationDiv);
}

function updateResults(elements, data) {
  elements.confidenceScore.textContent = data.confidence_score.toFixed(1);
  elements.reviewCount.textContent = data.total_reviews;
  if (data.sentiment_distribution) {
    const dist = data.sentiment_distribution;
    updateSentimentDisplay(elements, {
      positive: parseFloat(dist.positive) || 0,
      negative: parseFloat(dist.negative) || 0,
      neutral: parseFloat(dist.neutral) || 0
    });
  }
  elements.loadingState.classList.add('hidden');
  elements.results.classList.remove('hidden');
}

function updateSentimentDisplay(elements, distribution) {
  const container = document.createElement('div');
  container.className = 'stat-card sentiment-chart';
  container.innerHTML = `
    <div class="stat-label">Sentiment Distribution</div>
    <div class="sentiment-grid">
      <div>
        <div style="color: #22c55e">Positive</div>
        <div>${distribution.positive.toFixed(1)}%</div>
      </div>
      <div>
        <div style="color: #ef4444">Negative</div>
        <div>${distribution.negative.toFixed(1)}%</div>
      </div>
      <div>
        <div style="color: #6b7280">Neutral</div>
        <div>${distribution.neutral.toFixed(1)}%</div>
      </div>
    </div>
  `;
  // Clear previous results
  const existingChart = elements.results.querySelector('.sentiment-chart');
  if (existingChart) {
    existingChart.remove();
  }
  elements.results.appendChild(container);
}

// UI Update functions
function updateProgress(elements, percent, text) {
  elements.loadingText.textContent = text;
  elements.progressBar.style.width = `${percent}%`;
}

function createSentimentChart(distribution) {
  const container = document.createElement('div');
  container.className = 'stat-card';
  container.innerHTML = `
    <div class="stat-label">Sentiment Distribution</div>
    <div class="flex justify-between mt-2 text-sm">
      <div class="text-green-600">👍 ${distribution.positive}%</div>
      <div class="text-red-600">👎 ${distribution.negative}%</div>
      <div class="text-gray-600">😐 ${distribution.neutral}%</div>
    </div>
  `;
  document.getElementById('sentimentDistribution').appendChild(container);
}

function createFeatureHighlights(features) {
  const container = document.createElement('div');
  container.className = 'stat-card';
  let html = `<div class="stat-label">Top Discussed Features</div>`;
  features.forEach((feature, index) => {
    html += `
      <div class="mt-2 flex items-center justify-between ${index > 0 ? 'border-t pt-2' : ''}">
        <span class="text-sm">${feature.name}</span>
        <span class="text-xs px-2 py-1 rounded
          ${feature.sentiment === 'positive' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}">
          ${feature.mentions} mentions
        </span>
      </div>
    `;
  });
  container.innerHTML = html;
  document.getElementById('featureHighlights').appendChild(container);
}

function createRecommendation(recommendation) {
  const container = document.createElement('div');
  container.className = 'stat-card bg-blue-50 border-blue-200';
  container.innerHTML = `
    <div class="stat-label text-blue-800">Recommendation</div>
    <div class="mt-2 text-sm font-medium">${recommendation.verdict}</div>
    <div class="mt-1 text-xs text-blue-600">${recommendation.reasons.join(' ')}</div>
  `;
  document.getElementById('recommendation').appendChild(container);
}

function createVisualization(plotData) {
  const container = document.createElement('div');
  container.className = 'stat-card';
  container.innerHTML = `
    <div class="stat-label">Analysis Visualization</div>
    <img src="data:image/png;base64,${plotData}"
      alt="Sentiment Analysis Chart"
      class="mt-2 rounded">
  `;
  document.getElementById('visualization').appendChild(container);
}

// Initialization
document.addEventListener('DOMContentLoaded', () => {
  const elements = {
    analyzeBtn: document.getElementById('analyzeBtn'),
    loadingState: document.getElementById('loadingState'),
    loadingText: document.getElementById('loadingText'),
    progressBar: document.getElementById('progressBar'),
    results: document.getElementById('results'),
    confidenceScore: document.getElementById('confidenceScore'),
    reviewCount: document.getElementById('reviewCount')
  };

  chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
    document.getElementById('currentUrl').textContent =
      new URL(tabs[0].url).hostname;
  });

  elements.analyzeBtn.addEventListener('click', () => runAnalysis(elements));

  // Initialize extension state
  const initializeExtension = () => {
    const extensionState = {
      isDarkMode: localStorage.getItem('darkMode') === 'true',
      isAnalyzing: false,
      currentUrl: window.location.href
    };
    if (extensionState.isDarkMode) {
      document.body.classList.add('dark-mode');
    }
    return extensionState;
  };

  let state = initializeExtension();

  // Make extension persistent
  chrome.runtime.onConnect.addListener((port) => {
    port.onDisconnect.addListener(() => {
      keepAlive();
    });
  });

  function keepAlive() {
    setInterval(() => {
      chrome.runtime.connect({ name: 'keepAlive' });
    }, 20000);
  }

  // Dark mode toggle with animation
  document.getElementById('darkModeToggle').addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
  });

  // Enhanced analyze button with loading state
  document.getElementById('analyzeBtn').addEventListener('click', async () => {
    const button = document.getElementById('analyzeBtn');
    button.classList.add('analyzing');
    state.isAnalyzing = true;
    try {
      // Show progress animation
      document.getElementById('progressBar').style.width = '0%';
      document.getElementById('loadingState').classList.remove('hidden');
      // Animate progress
      let progress = 0;
      const progressInterval = setInterval(() => {
        progress += 1;
        document.getElementById('progressBar').style.width = `${progress}%`;
        if (progress >= 100) {
          clearInterval(progressInterval);
          // Removed analyzeLogic(); as it is undefined.
          runAnalysis(elements);
        }
      }, 50);
      
      // ...
    } catch (error) {
      console.error('Analysis failed:', error);
      button.classList.add('error');
    } finally {
      button.classList.remove('analyzing');
      state.isAnalyzing = false;
    }
  });

  // Close button with fade effect
  const closeBtn = document.getElementById('closeBtn');
  if (closeBtn) { // only add listener if element exists
    closeBtn.addEventListener('click', () => {
      const popup = document.getElementById('popupContainer');
      popup.classList.add('fade-out');
      setTimeout(() => {
        popup.style.display = 'none';
        popup.classList.remove('fade-out');
      }, 300);
    });
  }

  // Keep extension visible across navigation
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'pageChanged') {
      state.currentUrl = request.url;
      document.getElementById('currentUrl').textContent = state.currentUrl;
    }
  });

  // Initialize visibility
  document.getElementById('popupContainer').style.display = 'block';
  keepAlive();
});
function displayResults(data) {
    const resultsContainer = document.getElementById('resultsContainer');
    resultsContainer.innerHTML = '';
    data.forEach((result, index) => {
        const resultDiv = document.createElement('div');
        resultDiv.classList.add('result');
        resultDiv.innerHTML = `
            <h3>Review ${index + 1}</h3>
            <p><strong>Sentiment:</strong> ${result.sentiment}</p>
            <p><strong>Review:</strong> ${result.review}</p>
        `;
        resultsContainer.appendChild(resultDiv);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    let state = {
        isAnalyzing: false,
        currentPage: 1
    };
    
    async function analyzeProduct() {
        const button = document.getElementById('analyzeBtn');
        button.classList.add('analyzing');
        state.isAnalyzing = true;
        try {
            const currentUrl = document.getElementById('currentUrl').textContent;
            const apiUrl = getApiUrl(currentUrl);
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: currentUrl })
            });
            const data = await response.json();
            if (data.success) {
                displayResults(data.data);
            } else {
                throw new Error(data.error);
            }
        } catch (error) {
            console.error('Analysis failed:', error);
        } finally {
            button.classList.remove('analyzing');
            state.isAnalyzing = false;
        }
    }
    
    function displayResults(data) {
        document.getElementById('confidenceScore').textContent = data.confidence_score.toFixed(1);
        document.getElementById('reviewCount').textContent = data.total_reviews;
        document.getElementById('sentimentPlot').src = data.sentiment_plot;
        
        const featuresHtml = Object.entries(data.recommendations.positive_features)
            .map(([feature, count]) => `<div class="feature-item">${feature}: ${count}</div>`)
            .join('');
            
        document.getElementById('features').innerHTML = featuresHtml;
        
        document.getElementById('results').classList.remove('hidden');
    }
    
    document.getElementById('analyzeBtn').addEventListener('click', analyzeProduct);
 
  
});