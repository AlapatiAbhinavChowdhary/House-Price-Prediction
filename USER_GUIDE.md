# 🏠 How to Predict Your House Price - User Guide

## For New Users Who Want to Check Their House Price

### 📋 Prerequisites
- Python 3.8 or higher installed
- Internet connection (for downloading dependencies)

---

## 🚀 Step-by-Step Instructions

### Step 1: Get the Project
```bash
# Clone or download the project
git clone https://github.com/AlapatiAbhinavChowdhary/House-Price-Prediction.git
cd House-Price-Prediction
```

### Step 2: Install Required Packages
```bash
# Install all dependencies
pip install -r requirements.txt
```
⏱️ This takes about 2-3 minutes

### Step 3: Choose Your Method

---

## 🎯 Method 1: Use the Interactive Dashboard (EASIEST!)

### Launch the Dashboard
```bash
streamlit run app/streamlit_app.py
```

### Follow These Steps in the Dashboard:

1. **Wait for browser to open** (opens automatically at http://localhost:8501)

2. **In the sidebar, click "🔄 Load Data & Model"**
   - Wait for "✅ Data loaded successfully!" message
   - You'll see "✅ Model loaded successfully!" if a trained model exists

3. **If no model exists yet:**
   - The dashboard will show a warning
   - You need to train the model first (see Method 2 below)

4. **Navigate to "🔮 Predict" page** (in sidebar)

5. **Enter your house details:**
   - 💰 **Median Income**: Your area's median income (in $10,000s)
     - Example: If median income is $35,000, enter 3.5
   - 🏗️ **House Age**: How old is the house (in years)
   - 🛏️ **Avg Rooms**: Average number of rooms per household
   - 🛌 **Avg Bedrooms**: Average number of bedrooms per household
   - 👥 **Population**: Population in your block group
   - 👨‍👩‍👧‍👦 **Avg Occupancy**: Average household size
   - 🌍 **Latitude**: Your location's latitude
   - 🌎 **Longitude**: Your location's longitude

6. **Click "🔮 Predict Price"**

7. **See your predicted house price!** 💰

---

## 📊 Method 2: Train the Model First (One-Time Setup)

If you're the first user or the model hasn't been trained yet:

### Run the Training Notebook
```bash
jupyter notebook notebooks/complete_pipeline.ipynb
```

### In Jupyter:
1. Click "Cell" → "Run All" (or press Shift+Enter on each cell)
2. Wait for all cells to complete (~5-10 minutes)
3. The best model will be saved automatically
4. Now you can use Method 1 (Dashboard) for predictions!

---

## 💻 Method 3: Quick Python Script (For Developers)

### Create a simple prediction script:

```python
# predict_my_house.py
import sys
sys.path.append('src')

from data_loader import load_california_housing_data
from preprocessing import preprocess_pipeline
from feature_engineering import engineer_features
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('models/saved_models/best_model.pkl')
scaler = joblib.load('models/saved_models/scaler.pkl')
feature_names = joblib.load('models/saved_models/feature_names.pkl')

# YOUR HOUSE DETAILS - EDIT THESE VALUES
my_house = {
    'MedInc': 3.5,        # Median income in $10,000s
    'HouseAge': 25,       # House age in years
    'AveRooms': 5.0,      # Average rooms
    'AveBedrms': 1.0,     # Average bedrooms
    'Population': 1000,   # Block population
    'AveOccup': 3.0,      # Average occupancy
    'Latitude': 34.0,     # Your latitude
    'Longitude': -118.0   # Your longitude
}

# Create input dataframe
input_data = pd.DataFrame([my_house])

# Engineer features (same as training)
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
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]

# Display result
print("\n" + "="*50)
print("🏠 HOUSE PRICE PREDICTION")
print("="*50)
print(f"\n💰 Predicted Price: ${prediction * 100000:,.0f}")
print(f"\n📊 Price Breakdown:")
print(f"   - Price per Room: ${(prediction * 100000) / my_house['AveRooms']:,.0f}")
print(f"   - Price per Occupant: ${(prediction * 100000) / my_house['AveOccup']:,.0f}")
print(f"   - Income Multiplier: {(prediction * 100000) / (my_house['MedInc'] * 10000):.1f}x")
print("="*50 + "\n")
```

### Run it:
```bash
python predict_my_house.py
```

---

## 📍 How to Find Your Location Coordinates

### Option 1: Google Maps
1. Go to [Google Maps](https://maps.google.com)
2. Right-click on your location
3. Click the coordinates (e.g., "34.0522, -118.2437")
4. First number is Latitude, second is Longitude

### Option 2: GPS on Phone
- Use any GPS app
- Look for "Lat/Long" or "Coordinates"

### Option 3: Online Tools
- Visit [LatLong.net](https://www.latlong.net)
- Enter your address
- Get coordinates

---

## 💡 Understanding the Inputs

### Median Income (MedInc)
- **What it is**: Average income in your neighborhood
- **Format**: In $10,000s (e.g., $50,000 = 5.0)
- **Range**: Usually 0.5 to 15.0
- **Example**: If median income is $65,000, enter 6.5

### House Age
- **What it is**: How old the house is
- **Format**: Years
- **Range**: 1 to 52 years
- **Example**: Built in 2000, now 2026 = 26 years

### Average Rooms
- **What it is**: Average rooms per household in the area
- **Format**: Decimal number
- **Range**: Usually 2.0 to 10.0
- **Example**: 5.5 rooms

### Average Bedrooms
- **What it is**: Average bedrooms per household
- **Format**: Decimal number
- **Range**: Usually 0.5 to 5.0
- **Example**: 1.2 bedrooms

### Population
- **What it is**: Number of people in your block group
- **Format**: Whole number
- **Range**: Usually 100 to 5000
- **Example**: 1500 people

### Average Occupancy
- **What it is**: Average household size
- **Format**: Decimal number
- **Range**: Usually 1.0 to 6.0
- **Example**: 3.2 people per household

---

## ❓ Troubleshooting

### "No module named 'sklearn'"
```bash
pip install scikit-learn
```

### "Model file not found"
- You need to train the model first
- Run the Jupyter notebook (Method 2)
- Or download a pre-trained model (if available)

### "Dashboard won't open"
- Check if port 8501 is available
- Try: `streamlit run app/streamlit_app.py --server.port 8502`

### "Prediction seems wrong"
- Make sure income is in $10,000s (not dollars)
- Check that coordinates are for California
- Verify all inputs are reasonable values

---

## 🎯 Quick Example

**Scenario**: You want to predict price for a house in Los Angeles

```
Median Income: $60,000 → Enter: 6.0
House Age: 20 years → Enter: 20
Avg Rooms: 5.5 → Enter: 5.5
Avg Bedrooms: 1.2 → Enter: 1.2
Population: 1200 → Enter: 1200
Avg Occupancy: 3.0 → Enter: 3.0
Latitude: 34.0522 → Enter: 34.0522
Longitude: -118.2437 → Enter: -118.2437
```

**Click Predict** → Get your estimated house price! 🏠💰

---

## 📞 Need Help?

1. Check the main [README.md](README.md)
2. Review the documentation files
3. Open an issue on [GitHub](https://github.com/AlapatiAbhinavChowdhary/House-Price-Prediction/issues)

---

## 🎉 That's It!

You now know how to:
- ✅ Set up the project
- ✅ Train the model (if needed)
- ✅ Use the dashboard to predict prices
- ✅ Understand the input values
- ✅ Troubleshoot common issues

**Enjoy predicting house prices!** 🏠📈
