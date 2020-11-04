# Importing packages and functions
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import clean_data, get_features



# Setting random seed
RSEED = 42

# Reading the dataset
df = pd.read_csv("Home_Loan/application_train.csv.tar.gz")
print("Data successfully loaded.")

# Deleting the one line with all NaNs
df.dropna(subset=["TARGET"], inplace=True)

# Splitting the data with stratifying due imbalanced data
y = df.pop("TARGET")
X = df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify = y, random_state = RSEED)

# Saving the splitted test data
X_test.to_csv("Home_Loan/X_test.csv", index=False)
y_test.to_csv("Home_Loan/y_test.csv", index=False)
print("Test data successfully saved to Home_Loan.")

# Clean the data
X_train_cleaned, y_train_cleaned = clean_data(X_train, y_train, test=False)
print("Data successfully cleaned.")

# Sampling because of high imbalance
X_train_balanced, y_train_balanced = X_train_cleaned, y_train_cleaned
print("Data successfully balanced.")

# Selecting features
features = get_features()

# Training the model
model = LogisticRegression()
model.fit(X_train_balanced[features], y_train_balanced)
print("Model successfully fitted.")

# Saving the model
pickle.dump(model, open("models/model.sav", "wb"))

# Do predictions on train
y_train_pred = model.predict(X_train_balanced[features])
print("Prediction successful.")

# Print train results
print(classification_report(y_train_balanced, y_train_pred))
print(confusion_matrix(y_train_balanced, y_train_pred))