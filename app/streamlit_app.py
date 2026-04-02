# ============================================
# 📦 STEP 1: IMPORT LIBRARIES
# ============================================

import streamlit as st          # UI framework
import pickle                  # Load model & features
import numpy as np             # Numerical operations
import pandas as pd            # Data handling


# ============================================
# LOAD MODEL & FEATURE ORDER
# ============================================

# Load trained ML model
import os

BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "../model/rf_model_v1.pkl")
feature_path = os.path.join(BASE_DIR, "../model/features_v1.pkl")

model = pickle.load(open(model_path, "rb"))
feature_order = pickle.load(open(feature_path, "rb"))


# ============================================
# LOAD DATASET
# ============================================

import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)

data_path = os.path.join(BASE_DIR, "../data/events_sample2.csv")

print("DATA PATH:", data_path)   # debug
print("FILES:", os.listdir(os.path.join(BASE_DIR, "../data")))  # debug

df = pd.read_csv(data_path)

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract time features
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.dayofweek

# Map events to numeric
event_map = {'view': 0, 'addtocart': 1, 'transaction': 2}
df['event_score'] = df['event'].map(event_map)


# ============================================
#UI DESIGN
# ============================================

st.title("🛒 E-Commerce Recommendation & Purchase Prediction System")

st.write("This system recommends products and predicts purchase probability.")


# ============================================
# SELECT INPUT METHOD
# ============================================

option = st.radio("Choose Input Method", ["Manual Input", "Dataset User"])


# ============================================
#  GET USER FEATURES
# ============================================

if option == "Manual Input":

    # Manual input for demo
    hour = st.number_input("Hour (0-23)", 0, 23)
    day = st.number_input("Day (0-6)", 0, 6)
    total_events = st.number_input("Total Events", 1)
    avg_event_score = st.number_input("Avg Event Score", 0.0)

    purchase_count = st.number_input("Purchase Count", 0)
    view = st.number_input("View Count", 0)
    addtocart = st.number_input("Add to Cart Count", 0)
    transaction = st.number_input("Transaction Count", 0)

else:
    # Dataset-based input (realistic scenario)
    user_id = st.selectbox("Select User", df['visitorid'].unique())

    user_data = df[df['visitorid'] == user_id]

    # Extract meaningful features
    hour = int(user_data['hour'].mode()[0])
    day = int(user_data['day'].mode()[0])
    total_events = len(user_data)
    avg_event_score = user_data['event_score'].mean()

    purchase_count = (user_data['event'] == 'transaction').sum()
    view = (user_data['event'] == 'view').sum()
    addtocart = (user_data['event'] == 'addtocart').sum()
    transaction = purchase_count

    st.write(f"User stats → Events: {total_events}, Purchases: {purchase_count}")


# ============================================
# CREATE DERIVED FEATURES
# ============================================

# Ratio feature
view_to_cart_ratio = addtocart / (view + 1)

# Binary feature (whether user ever added to cart)
has_cart_activity = 1 if addtocart > 0 else 0


# ============================================
#  ITEM DATA (SIMULATION)
# ============================================

# Simulated item data (in real world → database)
item_data = pd.DataFrame({
    'itemid': [101, 102, 103, 104, 105],
    'item_popularity': [200, 150, 300, 180, 250],
    'item_engagement': [0.6, 0.4, 0.7, 0.5, 0.65]
})


# ============================================
# RUN RECOMMENDATION + PREDICTION
# ============================================

if st.button("🚀 Recommend & Predict"):

    results = []

    # Loop through recommended items
    for _, row in item_data.iterrows():

        # Prepare feature dictionary
        input_dict = {
            'hour': hour,
            'day': day,
            'total_events': total_events,
            'avg_event_score': avg_event_score,
            'purchase_count': purchase_count,
            'view': view,
            'addtocart': addtocart,
            'transaction': transaction,
            'view_to_cart_ratio': view_to_cart_ratio,
            'has_cart_activity': has_cart_activity,
            'item_popularity': row['item_popularity'],
            'item_engagement': row['item_engagement']
        }

        # Arrange features in correct order
        features = np.array(
            [input_dict[col] for col in feature_order]
        ).reshape(1, -1)

        # Predict probability
        prob = model.predict_proba(features)[0][1]

        # Apply threshold (IMPORTANT 🔥)
        prediction = 1 if prob > 0.5 else 0

        results.append((row['itemid'], prob, prediction))


    # ============================================
    # SORT RESULTS
    # ============================================

    results = sorted(results, key=lambda x: x[1], reverse=True)


    # ============================================
    # DISPLAY OUTPUT
    # ============================================

    st.subheader("🎯 Recommended Items")

    for item, prob, pred in results:

        st.write(f"🛍️ Item {item} → Probability: {prob:.2f}")

        if pred == 1:
            st.success("🔥 Likely to Buy")
        else:
            st.warning("⚠️ Less likely")