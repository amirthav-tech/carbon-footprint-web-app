# File: Data Preprocessing 

# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer
import ast 

# Load the dataset  
df = pd.read_csv("C:\Users\amirt\OneDrive\Desktop\carbonfootprint\data\Carbon Emission.csv")

# Fix NaN values in vehicle type column
df['Vehicle Type'] = df['Vehicle Type'].fillna('no_vehicle')

# Encode categorical features
categorical_features =[
    "Body Type","Sex","Diet","How Often Shower","Heating Energy Source","Transport","Vehicle Type",
    "Social Activity","Frequency of Traveling by Air","Waste Bag Size","Energy efficiency", "Recycling",
    "Cooking_With"
]  

# Label encoding (using mapping)
body_map = {'underweight': 0, 'normal': 1, 'overweight': 2, 'obese': 3}
diet_map = {'omnivore': 0, 'pescatarian': 1, 'vegetarian': 2, 'vegan': 3}
shower_map = {'less frequently': 0, 'daily': 1, 'more frequently': 2, 'twice a day': 3}
heating_map = {'coal': 0, 'wood': 1, 'natural gas': 2, 'electricity': 3}
transport_map = {'walk/bicycle': 0, 'public': 1, 'private': 2}
vehicle_type_map = {'no_vehicle':0,'petrol': 1, 'diesel': 3, 'lpg': 2, 'hybrid': 5, 'electric': 4}
waste_bag_size_map = {'small': 0, 'medium': 1, 'large': 2 ,'extra large':3}
travel_map = {'never': 0, 'rarely': 1, 'occasionally': 2, 'frequently': 3, 'very frequently': 4}

# Map categorical features to numerical values
df['Body Type'] = df['Body Type'].map(body_map)
df['Diet'] = df['Diet'].str.lower()  
df['Diet'] = df['Diet'].map(diet_map)
df['How Often Shower'] = df['How Often Shower'].map(shower_map)
df['Heating Energy Source'] = df['Heating Energy Source'].map(heating_map)
df['Transport'] = df['Transport'].map(transport_map)
df['Vehicle Type'] = df['Vehicle Type'].map(vehicle_type_map)  
df['Waste Bag Size'] = df['Waste Bag Size'].map(waste_bag_size_map)
df['Frequency of Traveling by Air'] = df['Frequency of Traveling by Air'].map(travel_map)

# Convert stringified lists into python lists
df['Recycling'] = df['Recycling'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
df['Cooking_With'] = df['Cooking_With'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# MultilabelBinarizer for Recycling and Cooking_With
mlb_recycle = MultiLabelBinarizer()
recycle_encoded = pd.DataFrame(mlb_recycle.fit_transform(df['Recycling']), columns=mlb_recycle.classes_)
df = pd.concat([df.drop('Recycling', axis=1), recycle_encoded], axis=1)

mlb_cooking = MultiLabelBinarizer()
cooking_encoded = pd.DataFrame(mlb_cooking.fit_transform(df['Cooking_With']), columns=mlb_cooking.classes_)
df = pd.concat([df.drop('Cooking_With', axis=1), cooking_encoded], axis=1)


# One Hot Encoding 
df = pd.get_dummies(df, columns=['Sex','Social Activity','Energy efficiency',
                    ],drop_first=True)


# Standardization of numerical features
numerical_features=[
    "Monthly Grocery Bill","Vehicle Monthly Distance Km","Waste Bag Weekly Count","How Long TV PC Daily Hour",
    "How Many New Clothes Monthly","How Long Internet Daily Hour"
]  
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Save the scaler
import joblib
joblib.dump(scaler, 'scaler.pkl')
scaler = joblib.load('scaler.pkl')
joblib.dump(mlb_recycle, 'mlb_recycle.pkl')
joblib.dump(mlb_cooking, 'mlb_cooking.pkl')
mlb_recycle = joblib.load('mlb_recycle.pkl')
mlb_cooking = joblib.load('mlb_cooking.pkl')




# Multicollinearity 
# There's no multicollinearity because two features have a correlation coefficient > 0.8 or < -0.8, 
# that could indicate multicollinearity . (From correlation matrix in 'eda_feature_engineering.ipynb' file.)

# Save the preprocessed data
df.to_csv('data/preprocessed_carbon_footprint_data.csv', index=False)
