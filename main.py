import numpy as np
from fontTools.ttx import process
from matplotlib.pyplot import xlabel
from pandas import Categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn import datasets
import  pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# لما تيجي تعمل baseline لا تعدل في فيتشرز ولا تشيل فيتشرز
# stage 1 : full intensive EDA ( understand data and its ranges and visualize data )

pd.options.display.max_rows = None
pd.options.display.max_columns = None

train_dataset_df = pd.read_csv('split/train.csv') # (1000000,10)
val_dataset_df = pd.read_csv('split/val.csv')
test_dataset_df = pd.read_csv('split/test.csv')

print("Training dataset information: \n",train_dataset_df.info())

print("Description of our training dataset: \n"
            ,tabulate( train_dataset_df.describe().T.round(2),headers='keys',tablefmt='fancy_grid')
            )

print(f"First 3 rows \n{train_dataset_df.head(3)}")

def prepare_data(train_dataset_df,val_dataset_df,test_dataset_df):
    train_dataset_df = train_dataset_df.drop('id', axis=1)

    float_train_dataset = train_dataset_df.select_dtypes(include=['float64'])
    int_train_dataset = train_dataset_df.select_dtypes(include=['int64'])
    num_train_dataset = train_dataset_df.select_dtypes(include=['int64', 'float64'])

    # القيم قريبة من بعضها وتابعة لل normal distribution ف  مش محتاجة يتعملها scale
    print(f"Describe of floating point numbers {float_train_dataset.describe()}")

    # calc skewness for each column
    print(f"Skewness \n {num_train_dataset.skew()}")  # 0 = symmetry, -ve = left skewed and +ve right skewed
    print(f"Int train dataset {int_train_dataset.describe()}")  # هنا أنا حابب أتفرج على ال statistics بتاع الداتا دي علشان الأرقام إلي ال Ranges (std) بتاعها كبير اعملها log
    # والقيم إلي ال scaled بتاعها صغير أعملها hash encoding or one hot encoding

    print(f"Sum of duplicated rows {train_dataset_df.duplicated().sum()}")  # 4 duplicated rows

    train_dataset_df.drop_duplicates(inplace=True)

    train_dataset_df['pickup_datetime'] = train_dataset_df['pickup_datetime'].astype('datetime64[ns]')

    # Create new features :
    train_dataset_df['Day_of_week'] = train_dataset_df.pickup_datetime.dt.dayofweek # monday,tuesday...
    train_dataset_df['Hour'] = train_dataset_df.pickup_datetime.dt.hour
    train_dataset_df['Month'] = train_dataset_df.pickup_datetime.dt.month
    train_dataset_df['Day_of_year'] = train_dataset_df.pickup_datetime.dt.dayofyear
    print(f"Describte trip duration column \n {train_dataset_df['trip_duration'].describe()}")  # Large scale values with outliers.


    train_dataset_df['trip_duration'] = np.log1p(train_dataset_df.trip_duration)
    # ال log1p مفيهاش أي مشكلة لإنها مش بتنقل اي statistics من ال train لل val

    # ال encoding مفيهاش اي مشكلة برضو لإنها مش بتنقل اي statistics


def approach1(train_dataset_df,val_dataset_df,test_dataset_df):
    Categorical_features = ['pickup_datetime','store_and_fwd_flag','Day_of_week','Day_of_year','Month']
    Numerical_features = ['passenger_count','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','Hour']
    train_features = Categorical_features + Numerical_features



# Stage 2 : Cleaning data understand each column ( missing values ,duplicate values and outliers )


# Stage 3 : Feature engineering ( encoding data , select features and add features )


# ML baseline model is a simple base model for our project to get a fast result.
# for incremental improves
# Stage 4 : Model

prepare_data(train_dataset_df,val_dataset_df, test_dataset_df)

# split data
Y = train_dataset_df[['trip_duration']]
train_dataset_df = train_dataset_df.drop(columns=['trip_duration'])
X = train_dataset_df
X_train,X_val,Y_train,Y_val = train_test_split(X,Y,test_size=0.2,shuffle=True,random_state=17)
print("featues train size ",X_train.shape," target train size ", Y_train.shape ," featues val size ",X_val.shape," target val size ", Y_val.shape )


def transform_train_val(X_train,X_val):
    processor = MinMaxScaler()
    X_train = processor.fit_transform(X_train)
    X_val = processor.transform(X_val)
    return X_train,X_val

X_train,X_val = transform_train_val(X_train,X_val)
Model = LinearRegression().fit(X_train,Y_train)

y_predicted_train = Model.predict(X_train)
print(f"r2 score for train is {r2_score(Y_train,y_predicted_train)} ")

y_predicted_val = Model.predict(X_val)
print(f"r2 score for val is {r2_score(Y_val,y_predicted_val)} ")


# # https://nyc-trip-duration.streamlit.app/
#
# import os
# import sys
# import pickle
# import numpy as np
# import pandas as pd
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel
# import io
# import pickle
#
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
# from src.Preprocessing.preprocessing import Preprocessing_Pipeline
# from src.Test.load import load_model
# from src.Enum.model_enums import ModelEnum as menum
#
# app = FastAPI(
#     title="NYC Trip Duration Prediction API",
#     version="1.0.0"
# )
#
# model_name = menum.XGBOOST.value
#
# model_path = f"src/{model_name}.pkl"
#
# with open(model_path, "rb") as f:
#     artifacts = pickle.load(f)
#
# model = artifacts['model']
# encode_season = artifacts['encode_season']
# encode_store = artifacts['encode_store']
# poly = artifacts['poly']
# scaler = artifacts['scaler']
#
# preprocess = Preprocessing_Pipeline()
#
# class TripInput(BaseModel):
#     vendor_id: int
#     pickup_datetime: str
#     passenger_count: int
#     pickup_longitude: float
#     pickup_latitude: float
#     dropoff_longitude: float
#     dropoff_latitude: float
#     store_and_fwd_flag: str
#
# @app.get("/")
# def health():
#     return {"status": "API is running"}
#
# @app.post("/predict")
# def predict_trip_duration(data: TripInput):
#
#     df = pd.DataFrame([data.model_dump()])
#     x = preprocess.transform(
#         df,
#         label_encoder_season=encode_season,
#         label_encoder_store=encode_store
#     )
#
#     x = poly.transform(x)
#     x = scaler.transform(x)
#     pred = model.predict(x)[0]
#     pred = np.exp(pred)
#
#     return {
#         "trip_duration_prediction": float(pred)
#     }
#
#
# @app.post("/predict_csv")
# def predict_trip_duration_csv(file: UploadFile = File(...)):
#
#     if not file.filename.endswith(".csv"):
#         raise HTTPException(status_code=400, detail="Only CSV files are supported")
#
#     try:
#         df = pd.read_csv(file.file)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Invalid CSV file: {e}")
#
#     try:
#         x = preprocess.transform(
#             df,
#             label_encoder_season=encode_season,
#             label_encoder_store=encode_store
#         )
#
#         x = poly.transform(x)
#         x = scaler.transform(x)
#
#         preds = model.predict(x)
#         preds = np.exp(preds)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#
#     df["trip_duration_prediction"] = preds
#
#     output = io.StringIO()
#     df.to_csv(output, index=False)
#     output.seek(0)
#
#     return StreamingResponse(
#         output,
#         media_type="text/csv",
#         headers={
#             "Content-Disposition": "attachment; filename=predictions.csv"
#         }
#     )
