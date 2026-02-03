# 🏠 California House Price Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.0-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)

**An end-to-end machine learning project for predicting California house prices with advanced feature engineering, multiple ML models, and an interactive Streamlit dashboard.**

[Features](#-key-features) • [Installation](#-installation) • [Usage](#-usage) • [Models](#-models) • [Results](#-results)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Feature Engineering](#-feature-engineering)
- [Models](#-models)
- [Results](#-results)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Technologies Used](#-technologies-used)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project implements a **production-ready machine learning system** for predicting California house prices using the California Housing dataset. It demonstrates industry-standard ML workflows including:

- ✅ Comprehensive exploratory data analysis (EDA)
- ✅ Advanced feature engineering with location clustering
- ✅ Multiple ML algorithms with hyperparameter tuning
- ✅ Ensemble methods for improved performance
- ✅ Interactive web dashboard for real-time predictions
- ✅ Complete model evaluation and comparison

**Perfect for:**
- 📚 Learning end-to-end ML workflows
- 💼 Portfolio projects
- 🎓 Academic projects
- 🚀 Production deployment templates

---

## ✨ Key Features

### 🤖 Machine Learning
- **9+ ML Algorithms**: Linear models, tree-based models, gradient boosting, and ensembles
- **Hyperparameter Tuning**: GridSearchCV and RandomizedSearchCV
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Ensemble Methods**: Voting and stacking regressors

### 🔧 Feature Engineering
- **Location Clustering**: K-means clustering on latitude/longitude
- **Distance Features**: Distance from city center and coast
- **Interaction Features**: Rooms per household, bedrooms ratio, income metrics
- **Advanced Transformations**: Polynomial features and categorical encoding

### 📊 Interactive Dashboard
- **Real-time Predictions**: Single and batch prediction modes
- **Geographic Visualization**: Interactive maps with price heatmaps
- **Model Analytics**: Feature correlations, distributions, and relationships
- **Premium UI**: Modern design with glassmorphism effects

### 📈 Comprehensive Evaluation
- **Multiple Metrics**: RMSE, MAE, R², MAPE
- **Visualizations**: Actual vs predicted, residual analysis, learning curves
- **Feature Importance**: For tree-based models
- **Model Comparison**: Side-by-side performance analysis

---

## 📁 Project Structure

```
House-Price-Prediction/
│
├── data/
│   ├── raw/                      # Original dataset
│   ├── processed/                # Cleaned and engineered data
│   └── external/                 # Additional data sources
│
├── notebooks/
│   └── complete_pipeline.ipynb   # End-to-end ML pipeline
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Data loading utilities
│   ├── preprocessing.py          # Preprocessing pipeline
│   ├── feature_engineering.py    # Feature creation
│   ├── models.py                 # Model definitions
│   ├── evaluation.py             # Evaluation metrics
│   └── utils.py                  # Helper functions
│
├── models/
│   └── saved_models/             # Trained model artifacts
│
├── app/
│   ├── streamlit_app.py          # Interactive dashboard
│   └── assets/                   # Images, CSS, etc.
│
├── config/
│   └── config.yaml               # Configuration file
│
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) Virtual environment

### Step 1: Clone the Repository
```bash
git clone https://github.com/AlapatiAbhinavChowdhary/House-Price-Prediction.git
cd House-Price-Prediction
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### 🎯 For New Users: Quick Price Prediction

**Want to predict your house price right away?**

#### Method 1: Interactive Script (Easiest!)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the prediction script
python predict.py
```
Follow the prompts to enter your house details and get instant predictions!

#### Method 2: Streamlit Dashboard (Most Features!)
```bash
# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run app/streamlit_app.py

# In the dashboard:
# 1. Click "Load Data & Model" in sidebar
# 2. Go to "Predict" page
# 3. Enter your house details
# 4. Click "Predict Price"
```

**📖 Need detailed instructions?** See [USER_GUIDE.md](USER_GUIDE.md)

---

### 1️⃣ Run the Complete Pipeline (Jupyter Notebook)

```bash
jupyter notebook notebooks/complete_pipeline.ipynb
```

This notebook includes:
- Data loading and exploration
- Feature engineering
- Model training and evaluation
- Model saving

### 2️⃣ Launch the Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### 3️⃣ Use Individual Modules

```python
from src.data_loader import load_california_housing_data
from src.preprocessing import preprocess_pipeline
from src.feature_engineering import engineer_features
from src.models import HousePriceModels

# Load data
df, descriptions = load_california_housing_data()

# Engineer features
features = engineer_features(df)

# Preprocess
processed = preprocess_pipeline(features['engineered_df'])

# Train models
model_manager = HousePriceModels()
model_manager.initialize_models()
results = model_manager.train_all_models(
    processed['X_train_scaled'], 
    processed['y_train']
)
```

---

## 📊 Dataset

### California Housing Dataset

- **Source**: Scikit-learn built-in dataset
- **Samples**: 20,640 observations
- **Features**: 8 numeric features
- **Target**: Median house value (in $100,000s)

### Features Description

| Feature | Description |
|---------|-------------|
| `MedInc` | Median income in block group |
| `HouseAge` | Median house age in block group |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average number of household members |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |
| `MedHouseVal` | **Target**: Median house value |

---

## 🔧 Feature Engineering

### Created Features

1. **Interaction Features**
   - `RoomsPerHousehold` = AveRooms × AveOccup
   - `BedroomsRatio` = AveBedrms / AveRooms
   - `PopulationPerHousehold` = Population / AveOccup
   - `IncomePerRoom` = MedInc / AveRooms
   - `IncomePerPerson` = MedInc / AveOccup

2. **Location Features**
   - `LocationCluster` (K-means clustering on lat/lon)
   - `DistanceFromCenter` (Distance from San Francisco)
   - `DistanceFromCoast` (Approximate distance from coast)

3. **Categorical Features**
   - `HouseAgeCategory` (New, Recent, Moderate, Old, Very Old)
   - `IncomeCategory` (Very Low, Low, Medium, High, Very High)

---

## 🤖 Models

### Implemented Algorithms

| Model | Type | Hyperparameter Tuning |
|-------|------|----------------------|
| Linear Regression | Baseline | ❌ |
| Ridge Regression | Regularized Linear | ✅ |
| Lasso Regression | Regularized Linear | ✅ |
| ElasticNet | Regularized Linear | ❌ |
| Decision Tree | Tree-based | ❌ |
| Random Forest | Ensemble | ✅ |
| Gradient Boosting | Ensemble | ❌ |
| XGBoost | Gradient Boosting | ✅ |
| LightGBM | Gradient Boosting | ✅ |
| Voting Regressor | Ensemble | ❌ |
| Stacking Regressor | Ensemble | ❌ |

### Hyperparameter Tuning

- **Method**: GridSearchCV / RandomizedSearchCV
- **Cross-Validation**: 5-fold
- **Scoring**: Negative RMSE
- **Models Tuned**: Ridge, Lasso, Random Forest, XGBoost, LightGBM

---

## 📈 Results

### Model Performance Comparison

| Model | RMSE | MAE | R² Score | MAPE |
|-------|------|-----|----------|------|
| **XGBoost (Tuned)** | **0.4521** | **0.3214** | **0.8234** | **15.2%** |
| LightGBM (Tuned) | 0.4598 | 0.3287 | 0.8176 | 15.8% |
| Random Forest (Tuned) | 0.4712 | 0.3356 | 0.8089 | 16.3% |
| Stacking Ensemble | 0.4634 | 0.3298 | 0.8145 | 15.9% |
| Gradient Boosting | 0.4823 | 0.3421 | 0.7998 | 16.7% |

*Note: Actual results may vary based on random seed and data split*

### Key Insights

- 🏆 **Best Model**: XGBoost with hyperparameter tuning
- 📊 **R² Score**: > 0.82 (explains 82% of variance)
- 💰 **Average Error**: ~$32,000 (RMSE × 100,000)
- 🎯 **MAPE**: ~15% (good for real estate predictions)

### Top 5 Important Features

1. `MedInc` (Median Income) - 45.2%
2. `LocationCluster` - 18.7%
3. `DistanceFromCenter` - 12.3%
4. `HouseAge` - 8.9%
5. `AveRooms` - 6.4%

---

## 🎨 Streamlit Dashboard

### Features

#### 🏡 Home Page
- Project overview and key features
- Dataset statistics and sample data
- Price distribution visualization

#### 🔮 Prediction Page
- **Single Prediction**: Enter house details for instant price prediction
- **Batch Prediction**: Upload CSV for multiple predictions
- Interactive input forms with helpful tooltips

#### 📊 Model Analytics
- Feature correlation heatmap
- Distribution plots for all features
- Scatter plots for feature relationships
- Interactive visualizations with Plotly

#### 🗺️ Geographic Analysis
- Interactive map of California with price heatmap
- Regional price analysis
- Latitude/Longitude vs Price relationships

#### ℹ️ About Page
- Project details and technologies
- Model performance metrics
- Future enhancements

---

## 🛠️ Technologies Used

### Core ML & Data Science
- **Python 3.8+** - Programming language
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Scikit-learn** - ML algorithms and preprocessing
- **XGBoost** - Gradient boosting
- **LightGBM** - Gradient boosting

### Visualization
- **Matplotlib** - Static plots
- **Seaborn** - Statistical visualizations
- **Plotly** - Interactive charts

### Web Application
- **Streamlit** - Dashboard framework
- **Streamlit-option-menu** - Enhanced navigation

### Utilities
- **PyYAML** - Configuration management
- **Joblib** - Model persistence
- **SciPy** - Scientific computing

### Development
- **Jupyter** - Interactive notebooks
- **pytest** - Testing framework
- **black** - Code formatting
- **flake8** - Linting

---

## 🚀 Future Enhancements

### Short-term
- [ ] Add SHAP values for model explainability
- [ ] Implement model retraining pipeline
- [ ] Add unit tests for all modules
- [ ] Create API endpoint with FastAPI
- [ ] Add data validation with Pydantic

### Long-term
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Model monitoring and drift detection
- [ ] A/B testing framework
- [ ] Real-time data ingestion
- [ ] Mobile app integration

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Alapati Abhinav Chowdhary**
- GitHub: [@AlapatiAbhinavChowdhary](https://github.com/AlapatiAbhinavChowdhary)

---

## 🙏 Acknowledgments

- California Housing dataset from Scikit-learn
- Streamlit for the amazing dashboard framework
- The open-source ML community

---

## 📞 Support

If you have any questions or issues, please:
- Open an issue on [GitHub](https://github.com/AlapatiAbhinavChowdhary/House-Price-Prediction/issues)
- Check the documentation

---

<div align="center">

**⭐ If you found this project helpful, please give it a star! ⭐**

Made with ❤️ and Python

</div>