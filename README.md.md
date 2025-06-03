# Phishing Website Detection Using Machine Learning

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Complete-success.svg" alt="Status">
</div>

## 📋 Overview

This project implements a machine learning-based approach to detect phishing websites using URL features. We compare three classification algorithms (Logistic Regression, SVM, and Random Forest) to identify the most effective solution for real-time phishing detection.

Key Achievement: 95.55% accuracy with Random Forest classifier using only URL-based features.

## 🎯 Problem Statement

- Phishing attacks account for 36% of all data breaches in 2023
- Average cost per incident: $4.91 million
- Traditional blacklist approaches fail against zero-day attacks
- Need for intelligent, scalable detection systems

## 🚀 Features

- Real-time Detection: Fast inference suitable for browser extensions
- High Accuracy: 95.55% accuracy with Random Forest
- Low Resource Usage: CPU-only, no GPU required
- Interpretable Results: Feature importance analysis for security analysts
- Scalable Architecture: Easy integration with existing security infrastructure

## 📊 Dataset

- Size: 95,910 websites
- Distribution: 58.3% legitimate, 41.7% phishing
- Features: 11 URL-based attributes
- Source: [Add dataset source if publicly available]

### Features Used

1. `activeDuration` - How long the domain has been active
2. `urlLen` - Length of the URL
3. `ranking` - Domain ranking (e.g., Alexa rank)
4. `domainLen` - Length of the domain name
5. `is_ip_address` - Whether URL uses IP address instead of domain
6. `count@` - Number of @ symbols in URL
7. `count.` - Number of dots in URL
8. `count-` - Number of hyphens in URL
9. `count_` - Number of underscores in URL
10. `count/` - Number of forward slashes in URL
11. `count?` - Number of question marks in URL

## 🔬 Methodology

### Algorithms Compared

1. Logistic Regression (Baseline)

   - Linear classification
   - Fast training (0.3s)
   - 88.18% accuracy

2. Support Vector Machine (SVM)

   - RBF kernel for non-linear patterns
   - Slower training (12.4s)
   - 90.99% accuracy

3. Random Forest (Best Performance)
   - 100 decision trees
   - Balanced speed (2.8s)
   - 95.55% accuracy

### Evaluation Metrics

| Model               | Accuracy | ROC-AUC | Training Time |
| ------------------- | -------- | ------- | ------------- |
| Logistic Regression | 88.18%   | 0.9457  | 0.3s          |
| SVM                 | 90.99%   | 0.9683  | 12.4s         |
| Random Forest       | 95.55%   | 0.9897  | 2.8s          |

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/phishing-detection.git
cd phishing-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
numpy==1.21.0
pandas==1.3.0
scikit-learn==0.24.2
matplotlib==3.4.2
seaborn==0.11.1
joblib==1.0.1
```

## 💻 Usage

### Training the Model

```python
from phishing_detector import PhishingDetector

# Initialize detector
detector = PhishingDetector(algorithm='random_forest')

# Train model
detector.train('data/phishing_dataset.csv')

# Save model
detector.save_model('models/phishing_rf_model.pkl')
```

### Making Predictions

```python
# Load trained model
detector = PhishingDetector.load('models/phishing_rf_model.pkl')

# Predict single URL
url = "http://suspicious-site.com/login"
is_phishing = detector.predict(url)
confidence = detector.predict_proba(url)

print(f"URL: {url}")
print(f"Phishing: {is_phishing}")
print(f"Confidence: {confidence:.2%}")
```

### Batch Processing

```python
# Predict multiple URLs
urls = [
    "https://www.legitimate-bank.com",
    "http://192.168.1.1/secure/login",
    "https://bit.ly/win-prize"
]

results = detector.predict_batch(urls)
for url, prediction in zip(urls, results):
    print(f"{url}: {'Phishing' if prediction else 'Legitimate'}")
```

## 📁 Project Structure

```
phishing-detection/
├── data/
│   ├── raw/                 # Original dataset
│   └── processed/          # Preprocessed data
├── models/
│   └── saved_models/       # Trained models
├── src/
│   ├── data_preprocessing.py
│   ├── feature_extraction.py
│   ├── model_training.py
│   └── phishing_detector.py
├── notebooks/
│   ├── EDA.ipynb          # Exploratory Data Analysis
│   └── Model_Comparison.ipynb
├── tests/
│   └── test_detector.py
├── docs/
│   └── presentation.pdf
├── requirements.txt
└── README.md
```

## 📈 Results Analysis

### Feature Importance (Random Forest)

1. activeDuration (32.1%) - Most important feature
2. ranking (18.5%)
3. urlLen (12.3%)
4. count. (9.7%)
5. is_ip_address (8.2%)

### Error Analysis

- False Positives (5.9%): Mainly legitimate sites using URL shorteners
- False Negatives (3.4%): Sophisticated attacks with homograph domains

## 🚧 Limitations

- Focuses only on URL features (no content analysis)
- May struggle with sophisticated homograph attacks
- Requires regular retraining for new phishing patterns
- Limited effectiveness on shortened URLs

## 🔮 Future Work

1. Enhanced Features

   - SSL certificate analysis
   - WHOIS metadata integration
   - DNS query patterns

2. Advanced Models

   - Deep learning approaches (CNN-LSTM)
   - Online learning capabilities
   - Ensemble methods with content analysis

3. Deployment
   - Browser extension development
   - REST API implementation
   - Real-time monitoring dashboard

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Dharmik Savani - [GitHub](https://github.com/dharmik097)
- Kirill Tatarnikov - [GitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- ISLA - Instituto Superior de Línguas e Administração
- Course: Inteligência Artificial - Engenharia de Tecnologias e Sistemas Web
- Date: June 4, 2025

## 📚 References

1. [Anti-Phishing Working Group Reports](https://apwg.org/)
2. [UCI Machine Learning Repository - Phishing Websites Dataset](https://archive.ics.uci.edu/ml/datasets/phishing+websites)
3. Random Forest Classification - Breiman, L. (2001)

---

<div align="center">
  <p>If you find this project useful, please consider giving it a ⭐</p>
</div>
