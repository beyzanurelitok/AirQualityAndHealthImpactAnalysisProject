import pandas as pd

# Dosya yolunu belirleyin

file_path = 'C:/anaconda/envs/HealthProject/HealthDataset/air_quality_health_impact_data.csv'

# CSV dosyasını oku
data = pd.read_csv(file_path)

# Verinin ilk iki satırını görüntüle
print(data.head(2))

#%%
#Gradient Algorithm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Specify the file path
file_path = 'C:/anaconda/envs/HealthProject/HealthDataset/air_quality_health_impact_data.csv'

# Read the CSV file
data = pd.read_csv(file_path)

# Function to classify HealthImpactScore into specific ranges
def classify_health_impact(score):
    if score >= 80:
        return 0
    elif score >= 60:
        return 1
    elif score >= 40:
        return 2
    elif score >= 20:
        return 3
    else:
        return 4

# Calculate HealthImpactScore and add HealthImpactClass column
data['HealthImpactClass'] = data['HealthImpactScore'].apply(classify_health_impact)

# Prepare the features (X) and target (y)
X = data[['AQI', 'PM10', 'PM2_5', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity'
          , 'WindSpeed', 'RespiratoryCases', 'CardiovascularCases', 'HospitalAdmissions']]
y = data['HealthImpactClass']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Gradient Boosting Classifier
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

# Function to get user input
def get_user_input():
    AQI = float(input("Enter AQI: "))
    PM10 = float(input("Enter PM10: "))
    PM2_5 = float(input("Enter PM2.5: "))
    NO2 = float(input("Enter NO2: "))
    SO2 = float(input("Enter SO2: "))
    O3 = float(input("Enter O3: "))
    Temperature = float(input("Enter Temperature: "))
    Humidity = float(input("Enter Humidity: "))
    WindSpeed = float(input("Enter Wind Speed: "))
    RespiratoryCases = int(input("Enter number of Respiratory Cases: "))
    CardiovascularCases = int(input("Enter number of Cardiovascular Cases: "))
    HospitalAdmissions = int(input("Enter number of Hospital Admissions: "))
    
    user_data = np.array([[AQI, PM10, PM2_5, NO2, SO2, O3, Temperature, Humidity
                           , WindSpeed, RespiratoryCases, CardiovascularCases, HospitalAdmissions]])
    return user_data

# Get user data
user_data = get_user_input()

# Scale user data
user_data_scaled = scaler.transform(user_data)

# Make a prediction
user_prediction = gb.predict(user_data_scaled)
user_prediction_proba = gb.predict_proba(user_data_scaled)

# Define class labels
health_impact_classes = ['Very High', 'High', 'Moderate', 'Low', 'Very Low']

# Convert prediction result to index and get class label
predicted_class_index = int(user_prediction[0])
predicted_class = health_impact_classes[predicted_class_index]
predicted_proba = user_prediction_proba[0][predicted_class_index] * 100

print(f'Predicted Health Impact Class: {predicted_class} ({predicted_proba:.2f}% confidence)')

# Evaluate the model
y_pred = gb.predict(X_test)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f'F1 Score: {f1}')
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

#%%
#Desicion Tree Algorithm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np


#read a file with pandas library
file_path = 'C:/anaconda/envs/HealthProject/HealthDataset/air_quality_health_impact_data.csv'

data = pd.read_csv(file_path)

#Find the header of data.
data.columns

#find the if there are null value in file
data.isnull().sum()

#split the values as features and target values
data_features = ['AQI', 'PM10', 'PM2_5', 'NO2', 'SO2', 'O3', 'Temperature',
       'Humidity', 'WindSpeed', 'RespiratoryCases', 'CardiovascularCases',
       'HospitalAdmissions']

data_target = ['HealthImpactClass']

X = data[data_features]
y = data[data_target]

#Split the value as train and test. Then decision that how much of it is training and how much of it is others, separate the file.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Apply standardscaler on values for take the good performance from data.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Determine that which one of the classification algorithms is better for this application
classifier = DecisionTreeClassifier()
classifier.fit(X_train_scaled, y_train)

y_pred = classifier.predict(X_test_scaled)

# f1_score hesaplama
f1 = f1_score(y_test, y_pred, average='weighted')
print("f1_score:", f1)

#Calculate the accuracy of algorithm.
accuracy = classifier.score(X_test_scaled, y_test)
print("Result of test:", accuracy)

#Write data of your have.
predict_value = [187.27005942368123,295.85303918510846,13.038560436938207,87.76956194227279,66.16114965083923,54.62427997507646,5.150335038375795, 84.42434365437401, 6.13775544737303,7,5,1]
predict_value = np.array(predict_value).reshape(1, -1)

predict_value_scaled = scaler.transform(predict_value)

# Tahmin yapma
predicted_result = classifier.predict(predict_value_scaled)
print("Girdi verilerinin sonucu:",predicted_result)

