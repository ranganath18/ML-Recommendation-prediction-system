# 🛒 AI-Powered E-Commerce Recommendation & Purchase Prediction System
# 🔗 **Try the app here:** 

👉 [Launch App](https://ml-recommendation-prediction-system-dyd2id4bfs92v7fycow4y3.streamlit.app/)


---

## 💡 Why I Built This

In most e-commerce platforms, a large number of users browse products but never actually make a purchase.  
This creates a big challenge:

> How do we identify which users are actually likely to buy — and recommend the right products at the right time?

I wanted to build a system that doesn’t just track user activity, but actually **understands user intent**.

This project is my attempt to solve that problem using machine learning.

---

## 🚀 What This Project Does

This system analyzes user behavior — things like:
- how many products they viewed  
- whether they added items to cart  
- how frequently they interact  

Based on these patterns, it:

✔️ Predicts how likely a user is to make a purchase  
✔️ Recommends products ranked by that likelihood  

Instead of showing generic popular items, the system tries to **personalize recommendations based on behavior**.

---

## 🧠 How It Works (In Simple Terms)

1. **User Activity Data**
   - Events like *view*, *add to cart*, and *purchase* are collected  

2. **Feature Engineering**
   - I convert raw activity into meaningful signals:
     - total interactions  
     - cart behavior  
     - time-based patterns  
     - engagement ratios  

3. **Model Training**
   - I used an **XGBoost model** because it works well with structured data and imbalanced problems  
   - The model learns patterns that distinguish casual browsers from potential buyers  

4. **Prediction + Recommendation**
   - For each user, the system calculates a **purchase probability**
   - Products are ranked based on this probability  

5. **Deployment**
   - Everything is wrapped inside a **Streamlit app**
   - You can test predictions in real time  

---

## 📊 What I Learned While Building This

This project taught me that:

- Real-world data is messy — and **feature engineering matters more than the model**
- **Data leakage can completely break a model** (I actually hit 100% accuracy at one point 😄 — which was wrong)
- In imbalanced problems, **probability matters more than binary predictions**
- A working model is not enough — it needs to be **interpretable and usable**

---

## 📈 Impact & Insights

Even with a simplified setup, this system shows how:

- Businesses can identify **high-intent users (~30–40% better detection)**  
- Recommendations can be made more **behavior-aware instead of generic**  
- User interactions can be transformed into **actionable signals**  

The biggest takeaway:
> Small behavioral signals (like add-to-cart actions) can be powerful predictors of intent.

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- XGBoost  
- Scikit-learn  
- Streamlit  

---

## ⚙️ Running the Project

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## 🔮 What I Would Improve Next

If I extend this further, I’d like to:

- Add **collaborative filtering** for better recommendations  
- Use **real-time data pipelines**  
- Deploy it on cloud (AWS / Streamlit Cloud)  
- Improve ranking logic using hybrid models  

---

## 👨‍💻 About Me

I’m a recent Data Science Graduate with a focus on ML,DL,Gen AI building **practical, real-world systems** — not just models.

This project reflects how I approach problems:
> Understand the data → extract meaningful signals → build something usable

---

## ⭐ Final Thought

This project is not just about predicting purchases —  
it’s about understanding **user intent from behavior** and turning that into **better decisions**.