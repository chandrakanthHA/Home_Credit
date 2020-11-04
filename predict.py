# Importing packages
import pandas as pd
import numpy as np
import pickle
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import clean_data, get_features



# Load model and data from disk
model = pickle.load(open("models/model.sav", 'rb'))
X_test = pd.read_csv("Home_Loan/X_test.csv")
y_test = pd.read_csv("Home_Loan/y_test.csv")
print("Model and data successfully loaded.")

# Cleaning the test data
X_test_cleaned, y_test_cleaned = clean_data(X_test, y_test, test=True)
print("Data successfully cleaned.")

# Selecting features
features = get_features()

# Do predictions on test
y_test_pred = model.predict(X_test_cleaned[features])
print("Prediction successful.")

# Print train results
print(classification_report(y_test_cleaned, y_test_pred))
print(confusion_matrix(y_test_cleaned, y_test_pred))