
# File : Model Training

import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression , Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score ,mean_absolute_error,root_mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import joblib 


# Load preprocessed dataset
df=pd.read_csv(r"C:\Users\amirt\OneDrive\Desktop\carbonfootprint\data\preprocessed_carbon_footprint_data.csv")

# Features and target 
X = df.drop(columns=['CarbonEmission'])
y = df['CarbonEmission']
joblib.dump(X.columns.tolist(), "models/training_columns.joblib")
training_columns = joblib.load("models/training_columns.joblib")

# Split the datset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Define models

models={
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.01),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'PolynomialRegression': make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        LinearRegression()
    )
}


# Training and Evaluation
results={}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)
    
    results[model_name] ={
        'MAE': round(mae,3),
        'MSE': round(mse,3),
        'RMSE': round(rmse,3),
        'R2 Score':round(r2,3)
    }
    
    # Save the model
    joblib.dump(model, f"models/{model_name}.joblib")
    
# Print results
print("Model Performance on Test Set:\n")
for model_name, metrics in results.items():
    print(f"{model_name}: {metrics}")