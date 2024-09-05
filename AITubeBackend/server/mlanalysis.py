# #bring in dependencies
# import joblib
# from fastapi import FastAPI
# from pydantic import BaseModel
# import pickle
# import pandas


# model_filename = 'completed_course_model.pkl'
# clf = joblib.load(model_filename)

# # with open('completed_course_model.pkl','rb') as f:
# #     model=pickle.load(f)



# app=FastAPI()

# class CoursePredictionInput(BaseModel):
#     age: int
#     gender: int
#     coursecount: int
#     time_spent: int
#     login_streak: int
#     score: int
#     codingsolved: int
#     skill_level: int


# @app.post("/predict")
# def predict_completion(data: dict):
#     input_data = data.get("data", {})  # Extract the 'data' dictionary
#     prediction = clf.predict([[
#         input_data["age"], input_data["gender"], input_data["coursecount"],
#         input_data["time_spent"], input_data["login_streak"],
#         input_data["score"], input_data["codingsolved"], input_data["skill_level"]
#     ]])
#     completion_status = "completed" if prediction[0] == 1 else "not completed"
#     return {"prediction": completion_status}



# import joblib
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from sklearn.preprocessing import LabelEncoder, StandardScaler

# scaler = StandardScaler()

# # Load the trained model
# model_filename = 'completed_course_model.pkl'
# model = joblib.load(model_filename)

# # Define the FastAPI app
# app = FastAPI()

# class InputData(BaseModel):
#     age: int
#     gender: int
#     coursecount: int
#     timespent: float
#     loginstreak: int
#     score: float
#     codingsolved: int
#     skilllevel: int

# @app.post("/predict")
# async def predict_completion(data: InputData):
#     try:
#         input_features = [
#             data.age, data.gender, data.coursecount, data.timespent,
#             data.loginstreak, data.score, data.codingsolved, data.skilllevel
#         ]
#         scaled_input_features = scaler.transform([input_features])
#         prediction = model.predict(scaled_input_features)
#         return {"prediction": bool(prediction[0])}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail="Prediction error")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# from fastapi import FastAPI
# from typing import Dict
# import pandas as pd
# import joblib
# from sklearn.preprocessing import LabelEncoder, StandardScaler

# app = FastAPI()

# # Load label encoders and scaler
# le_gender = LabelEncoder()
# le_skilllevel = LabelEncoder()
# scaler = StandardScaler()

# # Load the trained model
# model_filename = 'completed_course_model.pkl'
# clf = joblib.load(model_filename)

# # Preprocess example data
# # Fit label encoders on training data
# le_gender.fit(['Male', 'Female'])  # Replace with your actual gender categories
# le_skilllevel.fit(['Beginner', 'Intermediate', 'Advanced'])  # Replace with your actual skilllevel categories

# @app.post("/predict/")
# def predict_completion(data: Dict):
#     print(Dict)
#     example_data = {
#         'age': data['age'],
#         'gender': data['gender'],
#         'coursecount': data['coursecount'],
#         'timespent': data['timespent'],
#         'loginstreak': data['loginstreak'],
#         'score': data['score'],
#         'codingsolved': data['codingsolved'],
#         'skilllevel': data['skilllevel']
#     }

#     example_data['gender'] = le_gender.transform([example_data['gender']])[0]
#     example_data['skilllevel'] = le_skilllevel.transform([example_data['skilllevel']])[0]
#     example_features = ['age', 'gender', 'coursecount', 'timespent', 'loginstreak', 'score', 'codingsolved', 'skilllevel']
#     example_df = pd.DataFrame([example_data], columns=example_features)

#     # Fit the scaler on training data and transform example data
#     scaler.fit(X_train[example_features])  # Assuming X_train is the original training data
#     example_scaled = scaler.transform(example_df)

#     # Predict using the loaded model
#     prediction = clf.predict(example_scaled)

#     # Interpret the prediction
#     if prediction[0] == 1:
#         result = "Completed"
#     else:
#         result = "Not Completed"

#     return {"result": "hi"}

from fastapi import FastAPI
from typing import Dict
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = FastAPI()

# Load label encoders
le_gender = LabelEncoder()
le_skilllevel = LabelEncoder()

# Load the model and scaler
model_and_scaler_filename = 'completed_course_model_and_scaler.pkl'
model_and_scaler = joblib.load(model_and_scaler_filename)
clf = model_and_scaler['model']
scaler = model_and_scaler['scaler']

# Define Pydantic model for input data
class InputData(BaseModel):
    age: int
    gender: str
    coursecount: int
    timespent: int
    loginstreak: int
    score: int
    codingsolved: int
    skilllevel: str

# Fit label encoders on training data
le_gender.fit(['Male', 'Female'])  # Replace with your actual gender categories
le_skilllevel.fit(['Beginner', 'Intermediate', 'Advanced'])  # Replace with your actual skilllevel categories

@app.post("/predict/")
def predict_completion(data:InputData):
    data.dict()
    example_data = {
    "age": int(data.age),
    "gender": str(data.gender),
    "coursecount": int(data.coursecount),
    "timespent": int(data.timespent),
    "loginstreak": int(data.loginstreak),
    "score": int(data.score),
    "codingsolved": int(data.codingsolved),
    "skilllevel": str(data.skilllevel),
}

    example_data["gender"] = le_gender.transform([example_data["gender"]])[0]
    example_data["skilllevel"] = le_skilllevel.transform([example_data["skilllevel"]])[0]
    example_features = ["age", "gender", "coursecount", "timespent", "loginstreak", "score", "codingsolved", "skilllevel"]
    example_df = pd.DataFrame([example_data], columns=example_features)

    # Transform example data using the loaded scaler
    example_scaled = scaler.transform(example_df)

    # Predict using the loaded model
    prediction = clf.predict(example_scaled)

    # Interpret the prediction
    if prediction[0] == 1:
        result = "Completed"
    else:
        result = "Not Completed"

    return {"result": result}
