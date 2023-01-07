# Load the necessary libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.externals import joblib

# Load the patient data
# The data should be in a form such as:
#   [{'patient': {'age': 35, 'gender': 'female', 'medical_history': [{'disease': 'diabetes'}, ...]},
#     'medication': 'insulin',
#     'effectiveness': 'effective'},
#    {'patient': {...},
#     'medication': 'aspirin',
#     'effectiveness': 'not effective'},
#    ...]
patients = load_patient_data()

# Convert the patient data to a Pandas DataFrame
df = pd.DataFrame(patients)

# Define the input features and label
X = df[['age', 'gender', 'medical_history', 'medication']]
y = df['effectiveness']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Preprocessing pipeline
# This pipeline will impute missing values, one-hot encode categorical variables, and scale numerical variables
preprocessing = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ('scaler', StandardScaler())
])

# Define the preprocessing steps for each input feature
preprocessor = ColumnTransformer([
    ('age', StandardScaler(), ['age']),
    ('gender', OneHotEncoder(handle_unknown='ignore'), ['gender']),
    ('medical_history', preprocessing, ['medical_history']),
    ('medication', OneHotEncoder(handle_unknown='ignore'), ['medication'])
])

# Define the model
model = RandomForestClassifier()

# Create the full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# Define the hyperparameter search space
param_grid = {
    'model__n_estimators': [10, 100, 1000],
    'model__max_depth': [5, 10, None],
    'model__min_samples_split': [2, 5, 10]
}

# Create the grid search object
grid_search = GridSearchCV(pipeline, param_grid, cv=5)

# Fit the grid search object to the training data
grid_search.fit(X_train, y_train)

# Select the best model
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
f1 = f1_score(y_test, y_pred, average='micro')

# Print the evaluation metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1)

# Save the model to a file
model_file = 'medication_effectiveness_model.pkl'
joblib.dump(best_model, model_file)

