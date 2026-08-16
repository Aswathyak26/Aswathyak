import os
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. LOAD DATASET FROM EXCEL
print("Loading dataset...")
excel_path = 'dataset.xlsx'
xls = pd.ExcelFile(excel_path)
df1 = pd.read_excel(xls, sheet_name='eurovision_meta')

# 2. CLEANING & PREPROCESSING
# Drop unnecessary index column if present
if 'Unnamed: 0' in df1.columns:
    df1 = df1.drop(['Unnamed: 0'], axis=1)

# Remove duplicate rows and columns
df1 = df1.drop_duplicates()
df1 = df1.loc[:, ~df1.T.duplicated()]

# Handle Missing Continuous Numerical Features (Median)
continuous_cols = [
    'energy', 'duration', 'acousticness', 'danceability', 
    'tempo', 'speechiness', 'liveness', 'loudness', 'valence', 'Happiness'
]
for col in continuous_cols:
    if col in df1.columns:
        df1[col] = df1[col].fillna(df1[col].median())

# Handle Missing Discrete Numerical Features (Median/Mode)
discrete_cols = ['Semi.Final.Number', 'key', 'time_signature', 'mode']
for col in discrete_cols:
    if col in df1.columns:
        df1[col] = df1[col].fillna(df1[col].median())

# Handle Missing Categorical Features (Mode)
categorical_cols = ['Artist.gender', 'Group.Solo']
for col in categorical_cols:
    if col in df1.columns:
        df1[col] = df1[col].fillna(df1[col].mode()[0])

print("Data cleaning complete. Shape:", df1.shape)

# 3. SELECT FEATURES AND TARGET
features = [
    "Year", "Artist.gender", "Group.Solo", "Song.In.English", 
    "energy", "duration", "acousticness", "danceability", 
    "tempo", "speechiness", "key", "liveness", 
    "time_signature", "mode", "loudness", "valence", "Happiness"
]

x = df1[features]
y = df1["Points"]

# 4. ONE-HOT ENCODING
x_encoded = pd.get_dummies(
    x, 
    columns=['Artist.gender', 'Group.Solo', 'Song.In.English'], 
    dtype='int', 
    drop_first=True
)

# 5. TRAIN / TEST SPLIT
x_train, x_test, y_train, y_test = train_test_split(
    x_encoded, y, test_size=0.2, random_state=42
)

# 6. INITIALIZE & TRAIN GRADIENT BOOSTING REGRESSOR
print("Training Gradient Boosting Regressor...")
best_gb = GradientBoostingRegressor(
    subsample=0.6,
    n_estimators=500,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='log2',
    max_depth=8,
    learning_rate=0.01,
    random_state=42
)

best_gb.fit(x_train, y_train)

# 7. EVALUATE PERFORMANCE
y_pred = best_gb.predict(x_test)
print("=" * 50)
print(f"MAE : {mean_absolute_error(y_test, y_pred):.2f}")
print(f"MSE : {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²  : {r2_score(y_test, y_pred):.4f}")
print("=" * 50)

# 8. SAVE MODEL AND COLUMNS
joblib.dump(best_gb, 'eurovision_model.joblib')
joblib.dump(list(x_train.columns), 'model_columns.joblib')

print("Model and columns saved successfully!")