# Import require library
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load dataset
df = pd.read_csv('./data/engine_failure_dataset_synthetic.csv')

# Check for data types
print(df.dtypes)

df['Operational_Mode'].unique()

"""# Data cleaning"""

# check for duplicate
print(df.duplicated().sum())

# drop row with missing values
df.dropna(axis=0)

# convert Time_Stamp to datetime
df['Time_Stamp']  = pd.to_datetime(df['Time_Stamp'])

# df.drop('Operational_Mode', axis=1, inplace=True)
# df.corr()

"""# Exploratory Data analysis"""
print(df.describe())
# Distribution of Fault Condition
sns.countplot(x='Fault_Condition', data=df)
plt.title("Distribution of Fault Conditions")
plt.show()

# Visualization of Fault_Condition over time
plt.figure(figsize=(12, 4))
plt.plot(df['Time_Stamp'], df['Fault_Condition'])
plt.title("Fault Condition Over Time")
plt.xlabel("Time")
plt.ylabel("Fault Condition")
plt.show()

# Smoothed Fault Condition (Rolling Average)
df['Fault_Rolling'] = df['Fault_Condition'].rolling(window=50).mean()
# Visualize
plt.figure(figsize=(15, 4))
plt.plot(df['Time_Stamp'], df['Fault_Rolling'], color='orange')
plt.title("Smoothed Fault Condition Over Time (Rolling Average)")
plt.xlabel("Time")
plt.ylabel("Smoothed Fault Level")
plt.show()

"""# Data preprocessing"""
# Defining health status column by using moving average (50) e.g 0:Faulty, 1:Healthy
df['Health_Status'] = df['Fault_Rolling'].apply(lambda x: 'Faulty' if x > 1.5 else 'Healthy')
df['Health_Status'] = df['Health_Status'].map({'Healthy':1, 'Faulty':0})

# First replace the string 'NaN' with np.nan
print(df['Health_Status'].dtypes)
df.head()
df['Health_Status'].value_counts()

# split the df into X (features) and y (target)
X = df.drop(['Time_Stamp', 'Fault_Condition', 'Fault_Rolling', 'Health_Status'], axis=1)
y = df.iloc[:, 1:2].values

# Encode the categorical data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [8])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# split the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.2)

# # Scale features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

"""# Model Training"""
model = xgb.XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=200, subsample=1.0)
model.fit(X_train, y_train)

"""# Model Evaluation"""
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# pd.DataFrame()
# save encoder
joblib.dump(ct, filename='model/encoder.pkl')
# Save trained model
joblib.dump(model, filename='model/model.pkl')