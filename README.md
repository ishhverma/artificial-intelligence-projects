# artificial-intelligence-projects

Collection of small, self-contained Jupyter/Colab notebooks and lightweight scripts that demonstrate classic and deep-learning techniques across several tasks. This repository is intended as a set of reproducible learning examples you can run, extend, and adapt.

Permalink (this commit):
https://github.com/ishhverma/artificial-intelligence-projects/blob/9ae4d50cf97cfa2a54af5350367ac26b66abcf23/

Included projects / notebooks
- p1_Handwritten_Digit_Recognizer.ipynb — Handwritten Digit Recognition (MNIST) using MLP and CNN  
  https://github.com/ishhverma/artificial-intelligence-projects/blob/9ae4d50cf97cfa2a54af5350367ac26b66abcf23/p1_Handwritten_Digit_Recognizer.ipynb
- p2_Customer_Service_Chatbot.ipynb — FAQ-style customer service chatbot (NLTK + TF-IDF + cosine similarity)  
  https://github.com/ishhverma/artificial-intelligence-projects/blob/9ae4d50cf97cfa2a54af5350367ac26b66abcf23/p2_Customer_Service_Chatbot.ipynb
- p3_weather_forecasting.ipynb — Weather forecasting with LSTM (Meteostat data + feature engineering + LSTM)  
  https://github.com/ishhverma/artificial-intelligence-projects/blob/9ae4d50cf97cfa2a54af5350367ac26b66abcf23/p3_weather_forecasting.ipynb
- stockmarket_predict_p4.ipynb + train.py — Stock Market Prediction (Nifty50) — notebook experiments and a reproducible training script  
  https://github.com/ishhverma/artificial-intelligence-projects/blob/9ae4d50cf97cfa2a54af5350367ac26b66abcf23/stockmarket_predict_p4.ipynb

Table of Contents
- Overview
- p1: Handwritten Digit Recognizer (MNIST)
- p2: Customer Service Chatbot
- p3: Weather Forecasting with LSTM
- p4: Stock Market Prediction (Nifty50)
- Combined requirements
- How to run (Colab & locally)
- Troubleshooting & notes
- Suggested improvements & next steps
- Contributing, license & contact

---

Overview
This repository collects notebook-based experiments and a few scripts that show practical end-to-end workflows for learning and prototyping:
- Image classification (MNIST) with MLP/CNN and a Streamlit demo snippet.
- A classical NLP FAQ chatbot (NLTK + TF-IDF + cosine similarity).
- Time-series forecasting with LSTM using Meteostat weather data.
- A reproducible pipeline and experiments for stock market prediction (Nifty50) including tree-based models and neural networks.

These notebooks are educational reference material and not production systems. Use them as starting points for experiments, teaching, or rapid prototyping.

---

p1. Handwritten Digit Recognizer (MNIST)

What it contains
- p1_Handwritten_Digit_Recognizer.ipynb
  - Data loading (tf.keras.datasets.mnist)
  - Preprocessing, visualization, normalization, and label encoding
  - MLP and CNN model definitions, training, evaluation
  - Saving models in Keras format
  - Example Streamlit app snippet for image upload + prediction

Notes
- Typical example results: MLP test accuracy ~0.97–0.98, CNN ~0.99 (varies).
- Saved artifacts (if you execute the notebook): mlp_model.keras, cnn_model.keras

---

p2. Customer Service Chatbot

What it contains
- p2_Customer_Service_Chatbot.ipynb
  - Synthetic FAQ Q/A dataset and examples to expand it
  - NLTK-based preprocessing (tokenize, lemmatize, stopword filtering)
  - TF-IDF vectorization and cosine similarity intent matching
  - Regex-based entity extraction (e.g., order IDs)
  - Interactive loop example and example of saving objects with pickle

Behavior
- Preprocess and vectorize FAQ questions.
- For each user query, compute cosine similarity to find the best-matching FAQ.
- If similarity exceeds threshold, return an FAQ answer or an entity-aware response; otherwise prompt for clarification.

Notes
- Refit TF-IDF if you add or remove many questions.
- Colab input() has limitations; interactive loops run best locally or via a UI.

---

p3. Weather Forecasting with LSTM

What it contains
- p3_weather_forecasting.ipynb
  - Fetch daily weather data using Meteostat for a chosen location/date range
  - EDA, feature engineering (temporal, lag, rolling), scaling
  - Build look-back sequences and train an LSTM to predict next-day average temperature (tavg)
  - Save/visualize training history and predictions

Key steps
- Create lag (1–3 days) and 7-day rolling features for selected columns
- Drop NaNs after feature engineering (watch for accidental empty datasets)
- Scale features with MinMaxScaler and build (samples, look_back, n_features) arrays for LSTM

Notes
- Ensure Meteostat returns non-empty data for chosen coordinates and date range.
- Consider imputation instead of aggressive dropna to retain more samples.

---

p4. Stock Market Prediction (Nifty50)

What it contains
- stockmarket_predict_p4.ipynb — exploratory notebook with multiple model experiments (feature engineering, model comparisons including XGBoost, LightGBM, RandomForest, LSTM/GRU/CNN examples)
- train.py — reproducible script for the main pipeline: download Nifty50 historical data, engineer features, chronologically split, scale, and train LightGBM/XGBoost models (focus on reproducibility)
- requirements.txt — Python dependency list (if present)
- .gitignore and LICENSE (MIT)

Quickstart (script)
1. Create and activate a Python environment (python >= 3.8).
2. Install dependencies:
   pip install -r requirements.txt
3. Run:
   python train.py

Notes
- The notebook includes experiments with neural models; train.py focuses on a reproducible data pipeline and tree-based models (faster to run / reproduce).
- The script produces evaluation metrics and saved model artifacts (depending on implementation details in the notebook/script).

---

Combined requirements

Recommended Python: 3.8+

Common packages used across notebooks:
- numpy
- pandas
- matplotlib
- seaborn
- pillow (PIL)
- nltk
- scikit-learn
- tensorflow (tf.keras)
- meteostat (p3)
- xgboost, lightgbm (p4)
- streamlit (optional; p1 demo)
- pickle (builtin)

Install example:
```bash
pip install numpy pandas matplotlib seaborn pillow nltk scikit-learn tensorflow meteostat xgboost lightgbm streamlit
```

NLTK resource downloads (used by p2):
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
```

---

How to run

Google Colab (recommended for convenience/GPU)
- Open the target notebook in Colab using the GitHub → Open in Colab integration or the permalinks above.
- For MNIST training or LSTM training, enable GPU: Runtime → Change runtime type → GPU.
- Run cells top-to-bottom. Notebooks include pip/nltk install/download cells where needed.

Locally
1. Clone the repo:
   git clone https://github.com/ishhverma/artificial-intelligence-projects.git
   cd artificial-intelligence-projects
2. Create and activate a virtual environment, then install dependencies (see requirements.txt if present).
3. Open notebooks with Jupyter or run scripts (e.g., python train.py for p4).

Streamlit (MNIST demo)
- Save the example app code from p1 as app.py and ensure saved model file (cnn_model.keras) is present, then:
  streamlit run app.py

---

Troubleshooting & notes

NLTK LookupError
- If you see LookupError (e.g., punkt not found), run nltk.download('punkt') and restart the kernel.
- The chatbot notebook includes an extra `nltk.download('punkt_tab')` used to address an environment-specific issue in one run; typically `punkt` is enough.

TF-IDF / Chatbot
- After expanding the FAQ dataset, re-fit TfidfVectorizer and consider adjusting similarity_threshold (0.4–0.6).

Meteostat / Weather Notebook
- Ensure the location coordinates and date range yield data. If feature engineering + dropna empties the DataFrame, re-check fetched data.

Stock model reproducibility
- train.py focuses on deterministic preprocessing and training for reproducibility; check requirements for version pinning if exact results are needed.

TensorFlow / GPU
- Locally, ensure TensorFlow and CUDA/cuDNN versions are compatible for GPU training.

Pickle security
- Only unpickle files you trust; pickle can execute arbitrary code.

---

Contributing
- Open issues or PRs to:
  - Add requirements.txt and a LICENSE file
  - Move datasets into a `data/` directory
  - Add tests and CI
  - Add Dockerfiles for Streamlit demos and reproducible environments
  - Convert notebooks into modular scripts for easier reuse

If you want, I can:
- open a PR adding this README.md,
- add a requirements.txt with pinned versions,
- add a sample Streamlit app file for MNIST,
- or convert p2 into a small Streamlit/Flask app.

---

License & Author
Author: ishhverma  
Repository: https://github.com/ishhverma/artificial-intelligence-projects

Consider adding an explicit LICENSE file (MIT recommended) if you plan to publish this repository as open source.

Acknowledgements
- NLTK, scikit-learn, TensorFlow/Keras, Meteostat, XGBoost, LightGBM, Streamlit, and many open-source contributors.

Enjoy exploring and extending these notebooks!
