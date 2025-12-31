# Tuberculosis Risk Prediction using Logistic Regression

## 📋 Project Overview
A machine learning system that predicts tuberculosis (TB) risk based on demographic and clinical factors using Logistic Regression. This project demonstrates how predictive modeling can assist in medical screening by identifying high-risk individuals who may require further diagnostic testing.

## 🎯 Key Features
- **Clinical Risk Assessment**: Predicts TB probability using 6 key risk factors
- **Model Transparency**: Provides interpretable odds ratios for each feature
- **Performance Metrics**: Comprehensive evaluation with multiple metrics
- **Patient Risk Stratification**: Classifies patients as Low/Medium/High risk
- **Visual Analytics**: Essential visualizations for model interpretation

## 📊 Model Performance
The Logistic Regression model achieves:
- **Accuracy**: 85.83%
- **AUC Score**: 0.9125 (Excellent discriminatory power)
- **Precision**: 87.72%
- **Recall**: 83.33%
- **F1-Score**: 85.47%

## 🏥 Risk Factors Analyzed
The model analyzes 6 clinical/demographic features:

| Feature | Description | Risk Level |
|---------|-------------|------------|
| **Age** | Patient age in years | Higher risk with increasing age |
| **Sex** | Biological sex (0=Female, 1=Male) | Slightly higher risk for males |
| **Smoking Habits** | 0=Never, 1=Former, 2=Current | Most important risk factor |
| **Alcohol Consumption** | 0=Never, 1=Former, 2=Current | Second most important factor |
| **Diabetic Status** | 0=No, 1=Yes | Moderate risk increase |
| **Asthma History** | 0=No, 1=Yes | Slight risk increase |

## 🔬 How It Works

### Mathematical Foundation
The model uses logistic regression to calculate TB probability:

```
P(TB) = 1 / (1 + e^(-z))
Where: z = β₀ + β₁·Age + β₂·Sex + β₃·Smoking + β₄·Alcohol + β₅·Diabetes + β₆·Asthma
```

### Decision Rule
- **P(TB) ≥ 0.5** → Classify as "TB Patient"
- **P(TB) < 0.5** → Classify as "Control"

### Risk Stratification
- **Low Risk** (<30% probability): Routine monitoring
- **Medium Risk** (30-70%): Further investigation advised
- **High Risk** (>70%): Immediate medical attention required

## 📈 Key Insights from the Model

### Feature Importance (Odds Ratios)
1. **Smoking Habits** (OR ≈ 2.8): Most significant risk factor
2. **Alcohol Consumption** (OR ≈ 2.5): Second most important
3. **Age** (OR ≈ 1.8): Risk increases with age
4. **Diabetes** (OR ≈ 1.6): Moderate risk factor
5. **Asthma** (OR ≈ 1.4): Slight risk increase
6. **Sex** (OR ≈ 0.9): Slightly protective for females

### Clinical Validation
These findings align with established medical literature:
- Smoking damages lung mucosa, facilitating TB infection
- Alcohol weakens the immune system
- Diabetes impairs macrophage function
- Older age leads to immune senescence

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Required libraries: See `requirements.txt`

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/tb-risk-prediction.git
cd tb-risk-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 🚀 Usage

### Running the Complete Pipeline
```bash
python tb_predictor.py
```

### Code Structure
```python
# Initialize predictor
predictor = TuberculosisPredictor()

# Generate synthetic data
tb_data = predictor.generate_tb_data(n_samples=400)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
predictor.train_logistic_model(X_train, y_train)

# Evaluate model
metrics = predictor.evaluate_model(X_test, y_test)

# Make individual predictions
patient_features = [45, 1, 2, 1, 0, 1]  # Age, Sex, Smoking, Alcohol, Diabetes, Asthma
probability, prediction = predictor.predict_patient_risk(patient_features)
```

### Example Patient Cases
```python
# High-risk patient (97.3% probability)
[65, 1, 2, 2, 1, 1]  # Elderly male, smoker, drinker, diabetic, asthmatic

# Low-risk patient (14.2% probability)  
[25, 0, 0, 0, 0, 0]  # Young female, non-smoker, non-drinker, healthy

# Medium-risk patient (55.2% probability)
[45, 1, 1, 1, 1, 0]  # Middle-aged male, former smoker/drinker, diabetic
```

## 📊 Output Visualizations

The system generates three key visualizations:

### 1. Confusion Matrix
![Confusion Matrix]
<img width="693" height="607" alt="Screenshot 2025-12-31 221325" src="https://github.com/user-attachments/assets/09fd638c-51ba-4a3c-b4c8-d495fac2b655" />

- Shows true/false positives and negatives
- Visual representation of model accuracy

### 2. ROC Curve
![ROC Curve]
<img width="722" height="609" alt="Screenshot 2025-12-31 221332" src="https://github.com/user-attachments/assets/78072dee-0fb4-4ee2-bb67-90edbbf60317" />

- Displays trade-off between sensitivity and specificity
- AUC score indicates overall model performance

### 3. Feature Importance
![Feature Importance]
<img width="993" height="588" alt="Screenshot 2025-12-31 221336" src="https://github.com/user-attachments/assets/83452fbd-ce9c-4ff3-a816-1693e2434915" />

- Odds ratios show relative importance of each feature
- Helps understand what drives predictions

## 📋 Sample Output

```
TUBERCULOSIS PREDICTION USING LOGISTIC REGRESSION
==================================================

=== DATASET OVERVIEW ===
Total samples: 400
TB Patients: 200 (50.0%)
Controls: 200 (50.0%)

=== MODEL PERFORMANCE ===
Accuracy:    0.8583
Precision:   0.8772
Recall:      0.8333
F1-Score:    0.8547
AUC Score:   0.9125
Sensitivity: 0.8333
Specificity: 0.8833

=== PATIENT RISK ASSESSMENT ===
TB Probability: 0.973 (97.3%)
Prediction: TB PATIENT
Risk Level: HIGH RISK
Recommendation: Immediate medical attention required
```

## 🏥 Clinical Applications

### Primary Use Cases
1. **Screening Tool**: Identify high-risk individuals for targeted testing
2. **Resource Allocation**: Prioritize testing in resource-limited settings
3. **Patient Education**: Show patients their modifiable risk factors
4. **Public Health Planning**: Target interventions to high-risk populations

### Workflow Integration
```
Patient Data → Risk Score → Action
Low Risk (<30%) → Routine monitoring
Medium Risk (30-70%) → Schedule chest X-ray
High Risk (>70%) → Immediate sputum test + isolation
```

## 🔍 Model Limitations

### What This Model CAN Do:
- Identify statistical risk patterns
- Screen large populations quickly
- Provide interpretable risk factors
- Work with easily collectible data

### What This Model CANNOT Do:
- Diagnose active TB disease
- Replace microbiological confirmation
- Account for all clinical factors
- Handle rare or atypical presentations

### Important Considerations:
- **Simulated Data**: Uses synthetic data for demonstration
- **Binary Classification**: Only distinguishes TB vs Control
- **Screening Tool**: Not a diagnostic replacement
- **Complementary Role**: Should augment, not replace clinical judgment

## 📚 Methodology Details

### Data Generation
- 400 synthetic patient records
- Balanced dataset (200 TB, 200 Control)
- Risk factors based on epidemiological studies
- Realistic distributions matching clinical patterns

### Model Training
- **Algorithm**: Logistic Regression
- **Regularization**: L2 regularization (C=1.0)
- **Optimization**: liblinear solver
- **Validation**: 70/30 train-test split with stratification

### Evaluation Strategy
- Multiple metrics beyond accuracy
- Focus on clinical relevance (sensitivity/specificity)
- Visual model interpretation
- Individual case predictions

## 🧪 Testing & Validation

### Test Cases Included
```python
test_patients = [
    [65, 1, 2, 2, 1, 1],  # High-risk: Should predict TB
    [25, 0, 0, 0, 0, 0],  # Low-risk: Should predict Control
    [45, 1, 1, 1, 1, 0]   # Medium-risk: Borderline case
]
```

### Expected Outputs
1. **High-risk patient**: >70% probability, "HIGH RISK" classification
2. **Low-risk patient**: <30% probability, "LOW RISK" classification  
3. **Medium-risk patient**: 30-70% probability, "MEDIUM RISK" classification

## 📈 Performance Interpretation

### Model Strengths
1. **High AUC (0.9125)**: Excellent discriminatory power
2. **Good Balance**: Both sensitivity (83.3%) and specificity (88.3%)
3. **Clinical Relevance**: Features match known TB epidemiology
4. **Interpretability**: Clear odds ratios for clinical understanding

### Areas for Improvement
1. **Recall**: 16.7% of TB cases missed (could be improved)
2. **Feature Set**: Limited to 6 factors (could expand)
3. **Data Source**: Synthetic data (real data would be better)

## 🔮 Future Enhancements

### Planned Features
- [ ] Real clinical dataset integration
- [ ] Cross-validation implementation
- [ ] Hyperparameter tuning grid search
- [ ] Additional algorithms (Random Forest, XGBoost)
- [ ] Web interface for clinical use
- [ ] API endpoint for integration

### Research Directions
- Incorporate radiological features
- Add genetic risk factors
- Include geographical/epidemiological data
- Model latent vs active TB progression

**Note**: This project demonstrates machine learning concepts in healthcare. It uses synthetic data for educational purposes and should not be used for actual medical diagnosis or treatment decisions.
