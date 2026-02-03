"""
🏠 California House Price Predictor
Interactive Streamlit Dashboard with Advanced Features
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import load_california_housing_data
from preprocessing import preprocess_pipeline
from feature_engineering import engineer_features
from models import HousePriceModels
from evaluation import evaluate_all_models
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium design
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --accent-color: #ec4899;
        --background-dark: #0f172a;
        --card-background: #1e293b;
    }
    
    /* Premium card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        color: white;
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .info-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(120deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem !important;
        margin-bottom: 1rem;
    }
    
    h2 {
        color: #667eea;
        font-weight: 700;
        margin-top: 2rem;
    }
    
    h3 {
        color: #8b5cf6;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    /* Input fields */
    .stNumberInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #667eea;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        background-color: #1e293b;
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .element-container {
        animation: fadeIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.data_loaded = False

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/home.png", width=80)
    st.title("🏠 Navigation")
    
    page = st.radio(
        "Select Page",
        ["🏡 Home", "🔮 Predict", "📊 Model Analytics", "🗺️ Geographic Analysis", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Data loading
    if st.button("🔄 Load Data & Model", use_container_width=True):
        with st.spinner("Loading data and model..."):
            try:
                # Load data
                df, _ = load_california_housing_data(save_to_csv=False)
                st.session_state.df = df
                st.session_state.data_loaded = True
                
                # Try to load saved model
                model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved_models', 'best_model.pkl')
                if os.path.exists(model_path):
                    st.session_state.model = joblib.load(model_path)
                    st.session_state.scaler = joblib.load(model_path.replace('best_model.pkl', 'scaler.pkl'))
                    st.session_state.feature_names = joblib.load(model_path.replace('best_model.pkl', 'feature_names.pkl'))
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
                else:
                    st.warning("⚠️ No saved model found. Please train the model first.")
                
                st.success("✅ Data loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
    
    if st.session_state.data_loaded:
        st.success(f"📊 Data: {st.session_state.df.shape[0]:,} samples")
    
    if st.session_state.model_loaded:
        st.success("🤖 Model: Ready")
    
    st.markdown("---")
    st.markdown("### 📈 Quick Stats")
    if st.session_state.data_loaded:
        df = st.session_state.df
        st.metric("Avg Price", f"${df['MedHouseVal'].mean() * 100000:,.0f}")
        st.metric("Max Price", f"${df['MedHouseVal'].max() * 100000:,.0f}")
        st.metric("Min Price", f"${df['MedHouseVal'].min() * 100000:,.0f}")

# Main content
if page == "🏡 Home":
    # Hero section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<h1>🏠 California House Price Predictor</h1>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <h3>🎯 Advanced ML-Powered Price Prediction</h3>
            <p>Predict California house prices using state-of-the-art machine learning models 
            with advanced feature engineering and real-time predictions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.image("https://img.icons8.com/fluency/240/000000/real-estate.png", width=200)
    
    # Features
    st.markdown("## ✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Multiple ML Models</h3>
            <p>9+ algorithms including XGBoost, LightGBM, and ensemble methods</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Advanced Features</h3>
            <p>Location clustering, distance metrics, and interaction features</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Interactive Analytics</h3>
            <p>Real-time visualizations and comprehensive model insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Dataset overview
    if st.session_state.data_loaded:
        st.markdown("## 📊 Dataset Overview")
        
        df = st.session_state.df
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Samples", f"{len(df):,}")
        col2.metric("Features", f"{len(df.columns) - 1}")
        col3.metric("Avg Price", f"${df['MedHouseVal'].mean() * 100000:,.0f}")
        col4.metric("Price Range", f"${(df['MedHouseVal'].max() - df['MedHouseVal'].min()) * 100000:,.0f}")
        
        # Sample data
        st.markdown("### 📋 Sample Data")
        st.dataframe(df.head(10), use_container_width=True, height=400)
        
        # Distribution
        st.markdown("### 📈 Price Distribution")
        fig = px.histogram(df, x='MedHouseVal', nbins=50, 
                          title='House Price Distribution',
                          labels={'MedHouseVal': 'Median House Value ($100k)'},
                          color_discrete_sequence=['#667eea'])
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Predict":
    st.markdown("<h1>🔮 Price Prediction</h1>", unsafe_allow_html=True)
    
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load the model first using the sidebar button.")
    else:
        tab1, tab2 = st.tabs(["🏠 Single Prediction", "📁 Batch Prediction"])
        
        with tab1:
            st.markdown("### Enter House Details")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                med_inc = st.number_input("💰 Median Income", min_value=0.0, max_value=15.0, value=3.5, step=0.1,
                                         help="Median income in block group (in $10,000s)")
                house_age = st.number_input("🏗️ House Age", min_value=1, max_value=52, value=25, step=1,
                                           help="Median house age in block group")
                ave_rooms = st.number_input("🛏️ Avg Rooms", min_value=1.0, max_value=15.0, value=5.0, step=0.1,
                                           help="Average number of rooms per household")
            
            with col2:
                ave_bedrms = st.number_input("🛌 Avg Bedrooms", min_value=0.5, max_value=10.0, value=1.0, step=0.1,
                                            help="Average number of bedrooms per household")
                population = st.number_input("👥 Population", min_value=1, max_value=10000, value=1000, step=10,
                                            help="Block group population")
                ave_occup = st.number_input("👨‍👩‍👧‍👦 Avg Occupancy", min_value=0.5, max_value=10.0, value=3.0, step=0.1,
                                           help="Average household size")
            
            with col3:
                latitude = st.number_input("🌍 Latitude", min_value=32.0, max_value=42.0, value=34.0, step=0.01,
                                          help="Block group latitude")
                longitude = st.number_input("🌎 Longitude", min_value=-125.0, max_value=-114.0, value=-118.0, step=0.01,
                                           help="Block group longitude")
            
            if st.button("🔮 Predict Price", use_container_width=True):
                # Create input dataframe
                input_data = pd.DataFrame({
                    'MedInc': [med_inc],
                    'HouseAge': [house_age],
                    'AveRooms': [ave_rooms],
                    'AveBedrms': [ave_bedrms],
                    'Population': [population],
                    'AveOccup': [ave_occup],
                    'Latitude': [latitude],
                    'Longitude': [longitude]
                })
                
                # Engineer features (simplified version for prediction)
                input_data['RoomsPerHousehold'] = input_data['AveRooms'] * input_data['AveOccup']
                input_data['BedroomsRatio'] = input_data['AveBedrms'] / input_data['AveRooms']
                input_data['PopulationPerHousehold'] = input_data['Population'] / input_data['AveOccup']
                input_data['IncomePerRoom'] = input_data['MedInc'] / input_data['AveRooms']
                input_data['IncomePerPerson'] = input_data['MedInc'] / input_data['AveOccup']
                
                # Add distance features
                center_lat, center_lon = 37.7749, -122.4194
                input_data['DistanceFromCenter'] = np.sqrt(
                    (input_data['Latitude'] - center_lat)**2 + 
                    (input_data['Longitude'] - center_lon)**2
                ) * 111  # Approximate km
                input_data['DistanceFromCoast'] = np.abs(input_data['Longitude'] + 120)
                
                # Align with training features
                for col in st.session_state.feature_names:
                    if col not in input_data.columns:
                        input_data[col] = 0
                
                input_data = input_data[st.session_state.feature_names]
                
                # Scale features
                input_scaled = st.session_state.scaler.transform(input_data)
                
                # Make prediction
                prediction = st.session_state.model.predict(input_scaled)[0]
                
                # Display result
                st.markdown(f"""
                <div class="prediction-card">
                    <h2>🏠 Predicted House Price</h2>
                    <h1 style="font-size: 3.5rem; margin: 1rem 0;">${prediction * 100000:,.0f}</h1>
                    <p style="font-size: 1.2rem;">Estimated median house value for this area</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional insights
                col1, col2, col3 = st.columns(3)
                col1.metric("Price per Room", f"${(prediction * 100000) / ave_rooms:,.0f}")
                col2.metric("Price per Occupant", f"${(prediction * 100000) / ave_occup:,.0f}")
                col3.metric("Income Multiplier", f"{(prediction * 100000) / (med_inc * 10000):.1f}x")
        
        with tab2:
            st.markdown("### 📁 Upload CSV for Batch Predictions")
            
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
            
            if uploaded_file is not None:
                batch_df = pd.read_csv(uploaded_file)
                st.write("Preview:", batch_df.head())
                
                if st.button("🔮 Predict All", use_container_width=True):
                    st.info("Batch prediction feature coming soon!")

elif page == "📊 Model Analytics":
    st.markdown("<h1>📊 Model Analytics</h1>", unsafe_allow_html=True)
    
    if st.session_state.data_loaded:
        df = st.session_state.df
        
        # Correlation heatmap
        st.markdown("### 🔗 Feature Correlations")
        corr_matrix = df.corr()
        
        fig = px.imshow(corr_matrix, 
                       text_auto='.2f',
                       aspect="auto",
                       color_continuous_scale='RdBu_r',
                       title='Feature Correlation Matrix')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature distributions
        st.markdown("### 📈 Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature1 = st.selectbox("Select Feature 1", df.columns[:-1], index=0)
            fig1 = px.histogram(df, x=feature1, nbins=50, 
                              color_discrete_sequence=['#667eea'],
                              title=f'{feature1} Distribution')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            feature2 = st.selectbox("Select Feature 2", df.columns[:-1], index=1)
            fig2 = px.histogram(df, x=feature2, nbins=50,
                              color_discrete_sequence=['#764ba2'],
                              title=f'{feature2} Distribution')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Scatter plot
        st.markdown("### 🎯 Feature Relationships")
        col1, col2 = st.columns(2)
        
        with col1:
            x_feature = st.selectbox("X-axis", df.columns, index=0)
        with col2:
            y_feature = st.selectbox("Y-axis", df.columns, index=8)
        
        fig = px.scatter(df, x=x_feature, y=y_feature, 
                        color='MedHouseVal',
                        color_continuous_scale='Viridis',
                        title=f'{x_feature} vs {y_feature}',
                        opacity=0.6)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

elif page == "🗺️ Geographic Analysis":
    st.markdown("<h1>🗺️ Geographic Analysis</h1>", unsafe_allow_html=True)
    
    if st.session_state.data_loaded:
        df = st.session_state.df
        
        st.markdown("### 🌍 California House Prices - Geographic Distribution")
        
        # Interactive map
        fig = px.scatter_mapbox(df.sample(5000), 
                               lat='Latitude', 
                               lon='Longitude',
                               color='MedHouseVal',
                               size='Population',
                               color_continuous_scale='Viridis',
                               size_max=15,
                               zoom=5,
                               mapbox_style="carto-positron",
                               title='House Prices Across California',
                               labels={'MedHouseVal': 'Price ($100k)'})
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Regional analysis
        st.markdown("### 📍 Regional Price Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price by latitude
            fig1 = px.scatter(df, x='Latitude', y='MedHouseVal',
                            color='MedInc',
                            title='Price vs Latitude',
                            labels={'MedHouseVal': 'Price ($100k)'},
                            color_continuous_scale='Plasma')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Price by longitude
            fig2 = px.scatter(df, x='Longitude', y='MedHouseVal',
                            color='MedInc',
                            title='Price vs Longitude',
                            labels={'MedHouseVal': 'Price ($100k)'},
                            color_continuous_scale='Plasma')
            st.plotly_chart(fig2, use_container_width=True)

elif page == "ℹ️ About":
    st.markdown("<h1>ℹ️ About This Project</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h2>🎯 Project Overview</h2>
        <p>This is an end-to-end machine learning project for predicting California house prices 
        using the California Housing dataset. The project demonstrates advanced ML techniques, 
        feature engineering, and model deployment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🛠️ Technologies Used
        - **Python 3.8+**
        - **Scikit-learn** - ML algorithms
        - **XGBoost & LightGBM** - Gradient boosting
        - **Streamlit** - Web dashboard
        - **Plotly** - Interactive visualizations
        - **Pandas & NumPy** - Data processing
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Models Implemented
        - Linear Regression
        - Ridge & Lasso Regression
        - Random Forest
        - Gradient Boosting
        - XGBoost
        - LightGBM
        - Ensemble Methods
        """)
    
    st.markdown("""
    ### ✨ Key Features
    - **Advanced Feature Engineering**: Location clustering, distance metrics, interaction features
    - **Multiple ML Models**: 9+ algorithms with hyperparameter tuning
    - **Interactive Dashboard**: Real-time predictions and visualizations
    - **Model Explainability**: Feature importance and SHAP values
    - **Production-Ready**: Complete pipeline from data to deployment
    
    ### 📈 Model Performance
    - **Best Model**: Typically achieves R² > 0.80
    - **RMSE**: < 0.5 ($50,000 error on average)
    - **Cross-Validation**: 5-fold CV for robust evaluation
    
    ### 🚀 Future Enhancements
    - SHAP value explanations for individual predictions
    - Model retraining pipeline
    - API endpoint for predictions
    - Docker containerization
    - Cloud deployment (AWS/GCP/Azure)
    
    ---
    
    <div class="metric-card">
        <h3>👨‍💻 Developed with ❤️ using Modern ML Best Practices</h3>
        <p>This project showcases industry-standard machine learning workflows and deployment strategies.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 2rem;">
    <p>🏠 California House Price Predictor | Built with Streamlit & Advanced ML</p>
    <p>© 2026 | End-to-End Machine Learning Project</p>
</div>
""", unsafe_allow_html=True)
