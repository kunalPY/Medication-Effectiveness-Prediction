# Medication-Effectiveness-Prediction
This project aims to build a machine learning model that can predict how effective a medication will be for a particular patient based on their medical history and other factors.

## Requirements 
Python 3.5 or higher
scikit-learn 0.23 or higher
pandas 0.25 or higher

## Usage
```
[    {        'patient': {            'age': 35,            'gender': 'female',            'medical_history': [                {'disease': 'diabetes'},                {'disease': 'hypertension'}            ]
        },
        'medication': 'insulin',
        'effectiveness': 'effective'
    },
    {
        'patient': {
            'age': 55,
            'gender': 'male',
            'medical_history': [
                {'disease': 'asthma'},
                {'disease': 'allergy'}
            ]
        },
        'medication': 'aspirin',
        'effectiveness': 'not effective'
    },
    ...
]
```
To train and evaluate the model, you can use the `main`.py script:
```
python main.py /path/to/patient_data.json
```

This will output the evaluation metrics (accuracy, precision, recall, and F1 score) for the model on the test set.

You can also use the `predict.py` script to make predictions on new patient cases:
```
python predict.py /path/to/patient_data.json /path/to/new_patient_data.json
```
This will output the predicted effectiveness of the medication for each patient in the `new_patient_data.json` file.





