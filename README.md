# tubercluosis_prediction
Tuberculosis Prediction using Logistic Regression
A machine learning project that predicts tuberculosis (TB) risk using logistic regression based on clinical and demographic factors. This implementation follows clinical research methodologies and provides comprehensive model evaluation with essential medical visualizations.

📋 Project Overview
This project implements a logistic regression model to predict tuberculosis risk using six key clinical factors identified in tuberculosis research. The model provides:

Binary classification (TB Patient vs Control)

Probability-based risk assessment

Comprehensive model evaluation

Clinical decision support

Professional medical visualizations

🏥 Risk Factors Used
The model uses six clinically significant risk factors:

Age - Continuous variable (years)

Sex - Binary (0: Female, 1: Male)

Smoking Habits - Categorical (0: Never, 1: Former, 2: Current)

Alcohol Consumption - Categorical (0: Never, 1: Former, 2: Current)

Diabetic Status - Binary (0: No, 1: Yes)

History of Asthma - Binary (0: No, 1: Yes)

🚀 Features
📊 Model Evaluation
Confusion Matrix with performance metrics

ROC Curve with AUC score

Feature Importance using odds ratios

Comprehensive metrics: Accuracy, Precision, Recall, F1-Score, Sensitivity, Specificity

🎯 Clinical Applications
Individual patient risk assessment

Risk stratification (Low/Medium/High)

Clinical recommendations

Feature importance interpretation

📈 Visualizations
Feature distribution analysis

Confusion matrix with metrics

ROC curve with AUC

Odds ratios visualization

Performance comparison charts

🛠️ Installation
Prerequisites
Python 3.7+

pip package manager

Required Libraries
bash
pip install pandas numpy matplotlib seaborn scikit-learn
Installation Steps
Clone the repository:

bash
git clone https://github.com/yourusername/tuberculosis-prediction.git
cd tuberculosis-prediction
Install dependencies:

bash
pip install -r requirements.txt
Run the model:

bash
python tb_prediction.py
📁 Project Structure
text
tuberculosis-prediction/
│
├── tb_prediction.py          # Main implementation file
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── examples/                # Example usage scripts
│   └── clinical_cases.py    # Sample patient cases
└── outputs/                 # Generated graphs and results
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── feature_importance.png
💻 Usage
Basic Implementation
python
from tb_prediction import TuberculosisPredictor

# Initialize predictor
predictor = TuberculosisPredictor()

# Generate synthetic clinical data
data = predictor.generate_tb_data(400)

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
predictor.train_logistic_model(X_train, y_train)

# Evaluate model
metrics = predictor.evaluate_model(X_test, y_test)
Individual Patient Prediction
python
# Define patient features
patient_features = [45, 1, 2, 1, 1, 0]  # [age, sex, smoking, alcohol, diabetic, asthma]

# Get prediction
probability, prediction = predictor.predict_patient_risk(patient_features)

print(f"TB Probability: {probability:.3f}")
print(f"Risk Level: {'HIGH' if probability > 0.7 else 'MEDIUM' if probability > 0.3 else 'LOW'}")
Sample Patient Cases
Case	Age	Sex	Smoking	Alcohol	Diabetic	Asthma	Risk Level
High Risk	65	Male	Current	Current	Yes	Yes	HIGH
Medium Risk	45	Male	Former	Former	Yes	No	MEDIUM
Low Risk	25	Female	Never	Never	No	No	LOW
📊 Model Performance
Typical performance metrics from the implementation:

Metric	Value	Interpretation
Accuracy	~0.75	Good overall performance
AUC Score	~0.80	Excellent discrimination
Sensitivity	~0.72	Good at detecting TB cases
Specificity	~0.78	Good at identifying controls
Precision	~0.76	Reliable positive predictions
🎨 Visualizations
The project generates several key visualizations:

1. Confusion Matrix
<img width="737" height="560" alt="Screenshot 2025-10-11 124701" src="https://github.com/user-attachments/assets/8625d128-200a-410d-8d14-1aa4634890a7" />


Shows model prediction accuracy

Includes performance metrics

Color-coded for easy interpretation

2. ROC Curve
<img width="788" height="591" alt="image" src="https://github.com/user-attachments/assets/717da0ff-2927-450d-ba7c-10c068b136af" />


Displays model discrimination ability

Highlights AUC score

Compares against random classifier

3. Feature Importance
<img width="979" height="582" alt="image" src="https://github.com/user-attachments/assets/50be6e71-f0e6-4a51-b60b-223f9310ac0c" />


Shows odds ratios for each risk factor

Color-coded (green: risk factors, red: protective factors)

Clinical interpretation of feature importance

🏥 Clinical Interpretation
Risk Factor Analysis
Asthma History: Strongest predictor (Highest odds ratio)

Smoking Habits: Significant risk factor

Age: Moderate risk factor

Diabetes: Moderate association

Alcohol: Weaker association

Clinical Recommendations
Based on predicted probability:

< 0.3: Low risk - Routine monitoring

0.3-0.7: Medium risk - Further investigation

> 0.7: High risk - Immediate medical attention

🔬 Methodology
Data Generation
Synthetic clinical data with realistic distributions

Based on tuberculosis epidemiology research

Balanced dataset (50% TB patients, 50% controls)

Model Training
Algorithm: Logistic Regression

Preprocessing: StandardScaler for feature normalization

Validation: Train-test split (70-30)

Evaluation: Comprehensive medical metrics

Statistical Analysis
Odds ratios calculation

Confidence intervals

Feature significance testing

📈 Results Interpretation
Model Strengths
Good discriminatory power (AUC > 0.75)

Balanced sensitivity and specificity

Clinically interpretable features

Reliable risk stratification

Limitations
Synthetic data (real clinical data recommended for production)

Limited to six risk factors

Population-specific performance may vary

🚀 Future Enhancements
Planned improvements:

Integration with real clinical datasets

Additional risk factors (HIV status, malnutrition, etc.)

Advanced machine learning models (Random Forest, XGBoost)

Web interface for clinical use

API for integration with hospital systems

Mobile application for field workers

🤝 Contributing
We welcome contributions from the community! Areas for contribution:

Clinical Expertise: Improve risk factor selection

Data Science: Enhance model performance

Software Engineering: Code optimization and features

Medical Research: Validation with real datasets

Contribution Guidelines
Fork the repository

Create a feature branch

Make your changes

Add tests if applicable

Submit a pull request

📚 References
World Health Organization. (2023). Global Tuberculosis Report

Clinical research on TB risk factors and prediction models

Machine learning applications in medical diagnostics

Logistic regression in clinical prediction models
