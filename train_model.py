import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("loan_approval_dataset.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# -------------------------------
# 2. Data Preprocessing
# -------------------------------

# Drop loan_id column
df.drop("loan_id", axis=1, inplace=True)

# Encode categorical columns
le = LabelEncoder()

df["education"] = le.fit_transform(df["education"])
df["self_employed"] = le.fit_transform(df["self_employed"])
df["loan_status"] = le.fit_transform(df["loan_status"])

# -------------------------------
# 3. Separate Features and Target
# -------------------------------
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# -------------------------------
# 4. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 5. Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# -------------------------------
# 6. Train Model
# -------------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# -------------------------------
# 7. Save Model and Scaler
# -------------------------------
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully!")