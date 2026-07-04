"""
Parkinson's Disease Detection using XGBoost
=============================================
Yeh script UCI Parkinsons dataset use karke ek ML model banata hai
jo Parkinson's disease detect karta hai voice features ke basis par.

Is version mein model aur scaler ko FILE mein SAVE bhi kiya jata hai,
taaki baar baar train na karna pade. Ek baar train karo, phir
app.py (Streamlit web app) use karke turant predictions lo.

Requirements (pehle yeh install karo terminal/cmd mein):
    pip install numpy pandas scikit-learn xgboost joblib streamlit

Dataset:
    UCI ML Parkinsons Dataset (parkinsons.data)
    Download link: https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data
    Isse download karke apne project folder mein rakh lo.
"""

import os
import urllib.request
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier


# -----------------------------------------------------------------
# STEP 1: Dataset ko download karo (agar already nahi hai toh)
# -----------------------------------------------------------------
DATA_FILE = "parkinsons.data"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"

# Yeh naye files banenge - model aur scaler save karne ke liye
MODEL_FILE = "parkinsons_model.pkl"
SCALER_FILE = "parkinsons_scaler.pkl"
COLUMNS_FILE = "parkinsons_columns.pkl"


def download_dataset():
    if not os.path.exists(DATA_FILE):
        print("Dataset nahi mila, download kar raha hoon...")
        try:
            urllib.request.urlretrieve(DATA_URL, DATA_FILE)
            print("Dataset download ho gaya:", DATA_FILE)
        except Exception as e:
            print("Download fail ho gaya. Manually download karke isi folder mein daalo.")
            print("Link:", DATA_URL)
            print("Error:", e)
            exit()
    else:
        print("Dataset pehle se maujood hai:", DATA_FILE)


# -----------------------------------------------------------------
# STEP 2: Data load karo
# -----------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_FILE)
    print("\nPehle 5 records:")
    print(df.head())
    print("\nDataset shape (rows, columns):", df.shape)
    return df


# -----------------------------------------------------------------
# STEP 3: Features aur Labels alag karo
# -----------------------------------------------------------------
def get_features_labels(df):
    # 'name' column (index 0) aur 'status' column hata ke baaki sab features hain
    feature_columns = df.loc[:, df.columns != 'status'].columns[1:]  # column names save karenge
    features = df.loc[:, df.columns != 'status'].values[:, 1:]
    labels = df.loc[:, 'status'].values

    ones = labels[labels == 1].shape[0]
    zeros = labels[labels == 0].shape[0]
    print(f"\nLabel counts -> Disease (1): {ones}, Healthy (0): {zeros}")

    return features, labels, list(feature_columns)


# -----------------------------------------------------------------
# STEP 4: Features ko scale (normalize) karo
# -----------------------------------------------------------------
def scale_features(features):
    scaler = MinMaxScaler((-1, 1))
    x_scaled = scaler.fit_transform(features)
    return x_scaled, scaler


# -----------------------------------------------------------------
# STEP 5: Model train karo aur evaluate karo
# -----------------------------------------------------------------
def train_and_evaluate(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=7
    )

    model = XGBClassifier(eval_metric='logloss')
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred) * 100
    print(f"\n✅ Model Accuracy: {acc:.2f}%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model


# -----------------------------------------------------------------
# STEP 6: Model, Scaler aur Column names ko FILE mein SAVE karo
# -----------------------------------------------------------------
def save_everything(model, scaler, feature_columns):
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(feature_columns, COLUMNS_FILE)
    print(f"\n💾 Model save ho gaya: {MODEL_FILE}")
    print(f"💾 Scaler save ho gaya: {SCALER_FILE}")
    print(f"💾 Column names save ho gaye: {COLUMNS_FILE}")
    print("\nAb aap 'streamlit run app.py' chala ke web app use kar sakte ho!")


# -----------------------------------------------------------------
# MAIN — poora program yahan se chalega
# -----------------------------------------------------------------
if __name__ == "__main__":
    download_dataset()
    df = load_data()
    features, labels, feature_columns = get_features_labels(df)
    x_scaled, scaler = scale_features(features)
    model = train_and_evaluate(x_scaled, labels)
    save_everything(model, scaler, feature_columns)

    print("\n🎉 Program successfully complete ho gaya!")
