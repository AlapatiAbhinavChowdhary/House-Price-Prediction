"""
Simple House Price Prediction Script
For users who want to quickly predict their house price
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import joblib

def predict_house_price():
    """Interactive script to predict house price"""
    
    print("\n" + "="*60)
    print("🏠 CALIFORNIA HOUSE PRICE PREDICTOR")
    print("="*60)
    
    # Check if model exists
    model_path = 'models/saved_models/best_model.pkl'
    if not os.path.exists(model_path):
        print("\n❌ ERROR: Model not found!")
        print("\nYou need to train the model first:")
        print("1. Run: jupyter notebook notebooks/complete_pipeline.ipynb")
        print("2. Execute all cells in the notebook")
        print("3. Come back and run this script again")
        return
    
    # Load model and artifacts
    print("\n📦 Loading model...")
    model = joblib.load(model_path)
    scaler = joblib.load('models/saved_models/scaler.pkl')
    feature_names = joblib.load('models/saved_models/feature_names.pkl')
    print("✅ Model loaded successfully!")
    
    print("\n" + "="*60)
    print("📝 ENTER YOUR HOUSE DETAILS")
    print("="*60)
    print("\nTip: Press Enter to use default values shown in [brackets]\n")
    
    # Get user inputs
    try:
        med_inc = float(input("💰 Median Income (in $10,000s) [3.5]: ") or "3.5")
        house_age = float(input("🏗️  House Age (years) [25]: ") or "25")
        ave_rooms = float(input("🛏️  Average Rooms [5.0]: ") or "5.0")
        ave_bedrms = float(input("🛌 Average Bedrooms [1.0]: ") or "1.0")
        population = float(input("👥 Population [1000]: ") or "1000")
        ave_occup = float(input("👨‍👩‍👧‍👦 Average Occupancy [3.0]: ") or "3.0")
        latitude = float(input("🌍 Latitude [34.05]: ") or "34.05")
        longitude = float(input("🌎 Longitude [-118.24]: ") or "-118.24")
        
    except ValueError:
        print("\n❌ Invalid input! Please enter numeric values.")
        return
    
    # Create input data
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
    
    # Engineer features
    print("\n🔧 Engineering features...")
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
    ) * 111
    input_data['DistanceFromCoast'] = np.abs(input_data['Longitude'] + 120)
    
    # Align with training features
    for col in feature_names:
        if col not in input_data.columns:
            input_data[col] = 0
    
    input_data = input_data[feature_names]
    
    # Scale and predict
    print("🤖 Making prediction...")
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    
    # Display results
    print("\n" + "="*60)
    print("🎉 PREDICTION RESULTS")
    print("="*60)
    print(f"\n💰 PREDICTED HOUSE PRICE: ${prediction * 100000:,.0f}")
    print("\n📊 Additional Insights:")
    print(f"   • Price per Room:     ${(prediction * 100000) / ave_rooms:,.0f}")
    print(f"   • Price per Occupant: ${(prediction * 100000) / ave_occup:,.0f}")
    print(f"   • Income Multiplier:  {(prediction * 100000) / (med_inc * 10000):.1f}x")
    
    # Price range estimate (±10%)
    lower_bound = prediction * 0.9 * 100000
    upper_bound = prediction * 1.1 * 100000
    print(f"\n📈 Estimated Range (±10%):")
    print(f"   ${lower_bound:,.0f} - ${upper_bound:,.0f}")
    
    print("\n" + "="*60)
    print("\n✨ Prediction complete! Thank you for using the predictor.")
    print("\n💡 Tip: For more features, try the Streamlit dashboard:")
    print("   streamlit run app/streamlit_app.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        predict_house_price()
    except KeyboardInterrupt:
        print("\n\n👋 Prediction cancelled. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("\nPlease check:")
        print("1. Model files exist in models/saved_models/")
        print("2. All dependencies are installed (pip install -r requirements.txt)")
        print("3. You're running from the project root directory")
