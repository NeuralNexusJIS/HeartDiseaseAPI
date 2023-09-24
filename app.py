from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
app = FastAPI()

# Load the dataset
df = pd.read_csv('heart_disease_data.csv')

# X AND Y DATA
x = df.drop(['target'], axis=1)
y = df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Define a Pydantic model for input data
class UserData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int
    

# Initialize the machine learning model
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

@app.post("/predict/")
async def predict_diabetes(user_data: UserData):
    user_data_dict = user_data.dict()
    user_data_df = pd.DataFrame([user_data_dict])

    user_result = log_reg.predict(user_data_df)

    if user_result[0] == 0:
        output = 'Your Heart is Healthy'
    else:
        output = 'You have Heart Disease'

    # Calculate accuracy using the test data
    test_prediction = log_reg.predict(x_test)
    accuracy = accuracy_score(y_test, test_prediction) * 100

    return {
        "result": output,
        "accuracy": accuracy
    }
