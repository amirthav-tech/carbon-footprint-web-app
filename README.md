🌱 CarbonCurve - Carbon Footprint Predictor using Machine Learning
CarbonCurve is a machine learning-powered web application built with Streamlit and Polynomial Regression to estimate an individual's carbon footprint based on their lifestyle choices.
It not only predicts the footprint but also provides personalized recommendations to help reduce it.

📌 Features
ML Model: Polynomial Regression (outperformed Ridge, Lasso, Linear, and Random Forest)
Accurate Predictions: Uses preprocessed dataset from Kaggle's Individual Carbon Footprint Calculation.
User-Friendly UI: Nature-based, calm, and responsive design built in Streamlit.
Interactive Visuals: Plotly charts & metrics for better insights.
Personalized Recommendations: Lifestyle tips to reduce carbon footprint.
Profile & Data Saving: Users can create profiles and save their past results.
Vehicle Type Handling: Works even for users with walking/cycling habits (NaN handling in vehicle type).

📂 Project Structure
CarbonFootprint/
│

├── app/

 ├──app.py # Main Streamlit app
├── models/

 ├── model_train.py # Model training script
 
 ├── PolynomialRegression.joblib
 
 ├── LinearRegression.joblib
 
 ├── RidgeRegression.joblib
 
 ├── LassoRegression.joblib
 
 ├── RandomForestRegressor.joblib
├── preprocessing.py # Data preprocessing script

├── data/ # Dataset files ├──

├── requirements.txt # Python dependencies

└── README.md # Project documentation

🛠️ Tech Stack
Programming Language: Python
Libraries:
streamlit - Web app framework
scikit-learn - Machine learning model training
plotly - Interactive data visualizations
pandas, numpy - Data processing
joblib - Model persistence
Dataset: Individual Carbon Footprint Calculation - Kaggle

📊 Workflow
Data Preprocessing

Label encoding, one-hot encoding, and multilabel binarization for categorical features.
Scaling numerical features.
Handling NaN values for users who walk or cycle.
Model Training

Tested multiple regression models: Linear, Ridge, Lasso, Random Forest, and Polynomial Regression.
Polynomial Regression gave the best performance.
Deployment

Built an interactive Streamlit app for user input & predictions.
Visualized results using Plotly.
Deployed to Streamlit Cloud.
🚀 How to Run Locally
Clone the repository
git clone https:
cd carboncurve
Install dependencies
pip install -r requirements.txt
Run the app
streamlit run app.py

##🌍 How It Works

The user fills in lifestyle details (transport, energy usage, diet, etc.).

The app preprocesses these inputs using the same transformations applied during training.

The trained Polynomial Regression model predicts the carbon footprint.

The app displays:

Predicted Carbon Footprint (tons/year)
Impact Visualization

Personalized Recommendations
