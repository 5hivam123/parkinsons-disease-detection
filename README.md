# 🧠 Parkinson's Disease Detection using XGBoost

A Python Machine Learning project that detects the presence of Parkinson's disease in an individual based on biomedical voice measurements, using the **XGBoost** algorithm — with a training script and an interactive **Streamlit web app**.

---

## 📌 About the Project

Parkinson's disease is a progressive neurodegenerative disorder that affects movement and speech, caused by the degeneration of dopamine-producing neurons in the brain. Early detection is important because it allows for earlier treatment and better management of symptoms.

This project uses the **UCI ML Parkinsons Dataset**, which contains 195 voice recordings from 31 people (23 with Parkinson's disease). Each recording is represented by 22 biomedical voice measurements — things like average vocal frequency, jitter, shimmer, and noise-to-harmonics ratio — which tend to change when a person has Parkinson's.

We train an **XGBoost Classifier** on these features to predict whether a person has the disease (`status = 1`) or not (`status = 0`).

**Final model accuracy: ~95%**

---

## 🗂️ Project Structure

```
├── parkinsons_detection.py   # Trains the model, evaluates it, and saves it to disk
├── app.py                    # Streamlit web app for interactive predictions
├── parkinsons.data           # Dataset (downloaded automatically if missing)
├── parkinsons_model.pkl      # Saved trained model (generated after training)
├── parkinsons_scaler.pkl     # Saved MinMaxScaler (generated after training)
├── parkinsons_columns.pkl    # Saved feature column names (generated after training)
├── disease_photo.jpg         # (Optional) A photo you add yourself for the app UI
└── README.md                 # This file
```

---

## ⚙️ How It Works

1. **Data Loading** — Read the 195-record, 24-column UCI Parkinsons dataset with pandas.
2. **Feature/Label Split** — All columns except `name` and `status` are features; `status` is the label (0 = healthy, 1 = Parkinson's).
3. **Scaling** — Features are normalized to a range of -1 to 1 using `MinMaxScaler`, since XGBoost performs better on scaled data.
4. **Train/Test Split** — 80% of the data is used for training, 20% for testing.
5. **Model Training** — An `XGBClassifier` (gradient-boosted decision trees) is trained on the training set.
6. **Evaluation** — Accuracy, confusion matrix, and a classification report (precision/recall/F1) are printed.
7. **Persistence** — The trained model, scaler, and feature column names are saved to `.pkl` files with `joblib`, so the model doesn't need to be retrained every time.
8. **Web App** — `app.py` loads the saved model and lets a user enter voice-feature values (or load a sample) to get an instant prediction with a confidence score.

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
python -m pip install numpy pandas scikit-learn xgboost joblib streamlit
```

### 2. Train the model
```bash
python parkinsons_detection.py
```
This downloads the dataset (if not already present), trains the model, prints the accuracy/metrics, and saves the model files.

### 3. Run the web app
```bash
streamlit run app.py
```
This opens a browser window where you can enter feature values and get a prediction.

---

## 🩺 About Parkinson's Disease

Common symptoms include tremors, muscle stiffness, slowed movement (bradykinesia), balance problems, speech changes, smaller handwriting (micrographia), reduced facial expressions, and sleep disturbances. These symptoms usually develop gradually over several years. There is currently no cure, so early detection and consulting a neurologist is important for better long-term management.

> ⚕️ **Disclaimer:** This project is for educational purposes only and is **not** a substitute for professional medical diagnosis. Always consult a qualified doctor for any real health concern.

---

## 🧑‍🎓 What We Learned From This Project

- **End-to-end ML workflow** — how to go from a raw dataset to a working prediction system: load → clean/split → scale → train → evaluate → save → deploy.
- **Feature scaling matters** — using `MinMaxScaler` to normalize features improved how well the model could learn, since features here have very different ranges (e.g., frequency in Hz vs. small jitter percentages).
- **XGBoost basics** — how gradient boosting (an ensemble of decision trees, each correcting the errors of the previous one) can achieve high accuracy even on small datasets like this one (195 rows).
- **Model evaluation beyond accuracy** — accuracy alone isn't enough; the confusion matrix and classification report (precision, recall, F1-score) give a fuller picture of how well the model handles both classes, especially with an imbalanced dataset (147 disease cases vs. 48 healthy).
- **Model persistence** — how to save a trained model, its scaler, and feature metadata using `joblib`, so you don't have to retrain every time you want to make a prediction.
- **Building a simple ML web app** — how to use Streamlit to quickly wrap a trained model in an interactive UI that anyone can use, without needing to touch code or a Jupyter notebook.
- **Real-world debugging** — troubleshooting common environment issues along the way, such as missing `pip`, incompatible/experimental Python versions, and OneDrive's "cloud placeholder" files causing `FileNotFoundError` even when a file appears to exist.
- **Responsible AI/health messaging** — the importance of adding disclaimers and encouraging professional consultation when building any tool that touches on medical predictions.

---

## 📚 Dataset Credit

UCI Machine Learning Repository — Parkinsons Data Set
Source: https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data

---

## 🔮 Possible Future Improvements

- Add cross-validation and hyperparameter tuning (e.g., `GridSearchCV`) to squeeze out more accuracy.
- Try other algorithms (Random Forest, SVM, Neural Networks) and compare performance.
- Handle class imbalance more formally (e.g., SMOTE) since there are more disease cases than healthy ones.
- Deploy the Streamlit app online (e.g., Streamlit Community Cloud) so it's accessible without running it locally.
- Allow direct audio file upload, extracting the voice features automatically instead of manual entry.
