# Real Estate Price Prediction Model

## Overview
A machine learning model for predicting real estate prices using ensemble methods (Gradient Boosting and Random Forest). The model processes various property features including size, location, amenities, and crime rates to generate accurate price predictions.

## Authors & Collaborators
Viktoria Gagua
Tarang Kadyan
Maria Jose Aleman

## Key Features
- Comprehensive data preprocessing pipeline
- Feature engineering with domain-specific interactions
- Ensemble modeling combining Gradient Boosting and Random Forest
- Robust validation and diagnostic tools
- Outlier detection and handling
- Detailed performance metrics and visualizations

## Requirements
```
pandas>=1.3.0
numpy>=1.19.0
scikit-learn>=1.0.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Project Structure
```
├── train.csv               # Training dataset (not included)
├── test.csv                # Test dataset (not included)
├── Ciutat_Vella_Project.py # Main model implementation
└── submission.csv          # Model predictions (generated)
```

## Usage
1. Clone the repository:
```bash
git clone https://github.com/Viktoriag27/ML_real_estate_prediction.git
cd ML_real-estate-prediction
```

## Model Features
- Location-based features (neighborhood, crime rates)
- Property characteristics (size, rooms, bathrooms)
- Amenities (pool, AC, furnishing)
- Derived features (room density, crime density)

## Performance
- Cross-validated Mean RMSE: ~169
- Handles missing values and outliers
- Includes stability checks and diagnostics

---
**Note**: This repository is part of a real estate price prediction project. The training and test datasets are included in the repository.
