📚 Artificial Intelligence Projects
A curated collection of end-to-end AI & Machine Learning mini-projects built during my Artificial Intelligence Internship.

📌 Overview
This repository contains self-contained, reproducible Jupyter notebooks and scripts demonstrating practical workflows across multiple AI domains:
Computer Vision — MNIST digit recognition with MLP & CNN
Natural Language Processing — FAQ-style chatbot using NLTK & TF-IDF
Time-Series Forecasting — LSTM-based weather forecasting with Meteostat
Financial Modeling — Stock market prediction (Nifty50) with feature engineering + ML models
Each project includes data preparation, model development, evaluation, and notes for improvement.

🗂 Repository Structure
├── p1_Handwritten_Digit_Recognizer.ipynb
├── p2_Customer_Service_Chatbot.ipynb
├── p3_weather_forecasting.ipynb
├── stockmarket_predict_p4.ipynb
├── train.py   # Reproducible training pipeline for p4
├── README.md
└── (requirements.txt - recommended to add)

🚀 Included Projects
p1 — Handwritten Digit Recognizer (MNIST)
Tech: TensorFlow/Keras, CNN, MLP, Streamlit demo
Skills: image preprocessing, model evaluation, deployment snippet
Features
MNIST digit loading & visualization
Fully connected (MLP) and Convolutional Neural Network (CNN) models
Comparison of accuracy and training curves
Saved Keras models (mlp_model.keras, cnn_model.keras)
Optional Streamlit app for image prediction
🔧 Typical accuracy:
MLP: 97–98%
CNN: ≈99%

p2 — Customer Service Chatbot (NLP)
Tech: NLTK, TF-IDF, cosine similarity
Skills: text preprocessing, intent matching, lemmatization, stopword filtering
Features
Synthetic FAQ dataset with extendable Q/A pairs
Tokenization, lemmatization, stop-word removal
TF-IDF vectorization + cosine similarity ranking
Regex entity extraction (e.g., order IDs)
Interactive console loop
💡 Replaceable backend—can be upgraded to Sentence-BERT or spaCy NER.

p3 — Weather Forecasting with LSTM
Tech: LSTM, Meteostat, MinMax scaling, time-series engineering
Skills: data fetching, sliding windows, sequence modeling
Features
Automatic weather data download via Meteostat
Lag features (1–3 days)
Rolling averages
MinMax scaling for neural networks
LSTM to predict next-day average temperature
Training curves and prediction plots

p4 — Stock Market Prediction (Nifty50)
Tech: XGBoost, LightGBM, LSTM (experiments), feature engineering
Skills: forecasting, tree-model pipelines, reproducibility, scripts
Includes:
Notebook with extensive experimentation (LSTM, GRU, tree models, CNN trials)
train.py — reproducible pipeline for Nifty50 forecasting
Chronological splits (no leakage)
Feature engineering (technical indicators optional)
Model comparison metrics
🎯 Focus on efficiency & reproducibility for ML workflows.

🔧 Combined Requirements
Recommended Python: ≥ 3.8
numpy
pandas
matplotlib
seaborn
nltk
scikit-learn
tensorflow
xgboost
lightgbm
meteostat
streamlit (optional)
pillow

Install all at once:
pip install numpy pandas matplotlib seaborn pillow nltk scikit-learn tensorflow meteostat xgboost lightgbm streamlit

NLTK downloads:
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

▶️ How to Run
Google Colab (Recommended)
Open any notebook → Open in Colab
(Optional) Runtime → Change runtime type → GPU
Run top to bottom

🖥️ Local Execution
git clone https://github.com/ishhverma/artificial-intelligence-projects.git
cd artificial-intelligence-projects
pip install -r requirements.txt   # if added
Open Jupyter notebook:
jupyter notebook

Streamlit App (MNIST Demo)
Save demo code as app.py and run:
streamlit run app.py

Troubleshooting
NLTK LookupError
Install missing resources:
nltk.download('punkt')

Chatbot not matching queries
Refit TF-IDF after editing FAQ list
Adjust threshold (0.4–0.6 suggested)
Weather forecasting notebook empty after dropna
Reduce feature engineering or fetch a wider date range
TensorFlow GPU issues
Ensure CUDA/cuDNN versions match TensorFlow install.

🤝 Contributing
Feel free to open Issues or Pull Requests for:
Documentation improvements
Additional datasets
Better model architectures
Bug fixes or enhancements

📜 License
MIT License (recommended — add LICENSE file)

👤 Author
Ishh Verma
Artificial Intelligence Student / Intern
Repository: github.com/ishhverma/artificial-intelligence-projects
