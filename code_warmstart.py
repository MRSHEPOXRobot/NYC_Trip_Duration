from xmlrpc.client import Binary
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler,LabelEncoder,OrdinalEncoder
from sklearn.model_selection import  cross_val_score,GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor


pd.options.display.max_rows = None
pd.options.display.max_columns = None

def predict_eval(model, train, train_features, name):
    y_train_pred = model.predict(train[train_features])
    #rmse = mean_squared_error(train.log_trip_duration, y_train_pred, squared=False)
    rmse = mean_squared_error(train.log_trip_duration, y_train_pred)
    r2 = r2_score(train.log_trip_duration, y_train_pred)
    print(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}")


def approach1(train, test): # direct
    #numeric_features = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
    #categorical_features = ['dayofweek', 'month', 'hour', 'dayofyear', 'passenger_count']

    # Extracting all the categorical data to be encoded into numerical data
    binary_features = ['store_and_fwd_flag','vendor_id'] # Perform Ordinal Encoding عن طريق إن كل قيمة داخل الفيتشر تاخد فاليو خاصة بها ومترتبين حسب إنت عاوزهم يترتبوا إزاي
    numeric_features = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude','distance']
    cycling_features = ['dayofweek_sin','dayofweek_cos','hour_sin','hour_cos','dayofyear_sin','dayofyear_cos']

     # Don't encode the same concept twice : ال col الي اسمه month مش محتاج أدخله للموديل وأنا عندي كولن تاني زي ال dayofyear ممكن يديني نفس المعلومة وبسلاسة أكتر
    # وعلشان ميبقاش  في high correlation between dayofyear and month in data set

    # ال Label Encoding == Ordinal Encoding مع إختلاف إن الترتيب  ممكن في حتة ال Ordinal Encoding
    column_transformer = ColumnTransformer([ # الأعمدة الأصلية بعد ال Encoding تحذف
        ('binary',OrdinalEncoder(),binary_features), # ال label Encoding مش  موجود داخل ال ColumnTransformer ف بنستخدم ال OrdinalEncoder
        ('scaling', StandardScaler(), numeric_features) # ال scalling شغالة بس على ال Floating Point Numbers
        ]
        , remainder = 'passthrough'
    )

    print(f"After Encoding and scalling {train.head(5)}")

    pipeline1 = Pipeline(steps=[
        ('ohe', column_transformer),
        ('regression', Ridge(alpha=0.1,fit_intercept=True))
    ])
    pipeline2 = Pipeline(steps=[
        ('ohe', column_transformer),
        ('XGB', XGBRegressor(n_estimators=1000, learning_rate=0.05, gamma=0, subsample=0.75, max_depth=7, random_state=5,
                 min_child_weight=1, colsample_bytree=0.8))
    ])
    pipeline3 = Pipeline(steps=[
        ('ohe', column_transformer),
        ('GBR',
         GradientBoostingRegressor(random_state=5, n_estimators=1000, learning_rate=0.05,
                                   max_depth=9, min_samples_split=17, max_features='sqrt',
                                   min_samples_leaf=13, loss='huber'))
         ])


    def modelfit(alg, train, target, test, target_test, performCV=True, printFeatureImportance=True, cv_folds=5):
        # Fit the algorithm on the data
        alg.fit(train, target)

        # Predict training set:
        train_predictions = alg.predict(train)
        test_predictions = alg.predict(test)

        # Perform cross-validation:
        if performCV:
            cv_score = cross_val_score(alg, train, target, cv=cv_folds)

        # Print model report:
        print("\nModel Report")
        print("Accuracy on train: {}".format(alg.score(train, target)))
        print("Accuracy on test: {}".format(alg.score(test, target_test)))
        print("Mean absolute error : {}".format(mean_absolute_error(target_test, test_predictions)))
        print("Mean squared error : {}".format(mean_squared_error(target_test, test_predictions)))
        print("Root mean squared error : {}".format(np.sqrt(mean_squared_error(target_test, test_predictions))))

        if performCV:
            print("CV Score : Mean - {} | Std - {} | Min - {} | Max - {}".format(np.mean(cv_score), np.std(cv_score),
                                                                                 np.min(cv_score), np.max(cv_score)))

            # Print Feature Importance:
            if printFeatureImportance:
                feature_imp = alg.feature_importances_.tolist()
                feature_columns = list(train.columns)
                Import_Df = pd.DataFrame({'Feature': list(train.columns),
                                          'Importance': alg.feature_importances_})
                Import_Df = Import_Df.sort_values(by='Importance', ascending=False)

                fig = plt.figure(figsize=(15, 10))
                fig = sns.barplot(data=Import_Df, x='Feature', y='Importance')
                plt.xticks(rotation=90)
                plt.title("Feature Importances", fontsize=20)
                plt.xlabel("Feature", fontsize=15)
                plt.ylabel("Importances", fontsize=15)
                plt.show()
    # Try another ML Model

    #xgb0 = XGBRegressor(n_estimators=1000, learning_rate=0.05, gamma=0, subsample=0.75, max_depth=7, random_state=5,
      #                  min_child_weight=1, colsample_bytree=0.8)
    #modelfit(xgb0, X_train, y_train, X_test, y_test)

    #gbm0 = GradientBoostingRegressor(random_state=5)
    #modelfit(gbm0, X_train, y_train, X_test, y_test)

    train_features = numeric_features + binary_features + cycling_features

    model = pipeline1.fit(train[train_features], train.log_trip_duration)
    predict_eval(model, train, train_features, "train")
    predict_eval(model, test, train_features, "test")


def prepare_data(train):
    train.drop_duplicates(inplace=True)
    train.drop(columns=['id'], inplace=True)

    # في حلين للتعامل مع الأعمدة ال timestamp وهو إني أغير النوع بتاعها astype('int64')  مثلاً أو إني أعملها feature Extraction
    train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
    # pickup_datetime ==> feature extraction then delete this column.
    train['dayofweek'] = train.pickup_datetime.dt.dayofweek
    train['hour'] = train.pickup_datetime.dt.hour
    train['dayofyear'] = train.pickup_datetime.dt.dayofyear

    # Calc distance using Manhattan distance formula.
    train['distance'] = (
        np.abs(train.pickup_longitude - train.dropoff_longitude)
        + np.abs(train.pickup_latitude - train.dropoff_latitude)    )

    train['log_trip_duration'] = np.log1p(train.trip_duration)



    # أي feature زمني ودوري
    # → Cyclical Encoding (sin & cos)
    def cycling_encoding(train, col, max_val):
        train[col + '_sin'] = np.sin(2 * np.pi * train[col] / max_val)
        train[col + '_cos'] = np.cos(2 * np.pi * train[col] / max_val)
        return train

    # بيحوّل الزمن لنقطة على دايرة بالتالي  الموديل يفهم القرب الزمني الحقيقي  العلاقات الدورية تبقى واضحة
    train = cycling_encoding(train, 'hour', 24)
    train = cycling_encoding(train, 'dayofweek', 7)
    train = cycling_encoding(train, 'dayofyear', 365)
    # ترك العمودين يعني أنك تعطي الموديل نفس المعلومة مرتين ف لازم تمسح الأعمدة الأصلية Multicollinearity
    train.drop(['dayofweek', 'hour', 'trip_duration', 'pickup_datetime', 'dayofyear'], axis=1, inplace=True)

    print(f"Before Encoding \n {train.head(5)}")

def visualization(train):
    print(train.describe())

if __name__ == '__main__':
    #root_dir = 'project-nyc-taxi-trip-duration'
    #train = pd.read_csv(os.path.join(root_dir, 'split_sample/train.csv'))
    train = pd.read_csv('src/data/train.csv')
    #test = pd.read_csv(os.path.join(root_dir, 'split_sample/val.csv'))
    test = pd.read_csv('src/data/val.csv')

    prepare_data(train)
    prepare_data(test)

    visualization(train)

    approach1(train, test)

# الموديل في الأول كان جاب القيم العالية دي في التران واوفرفت  في الفال بسبب الفيتشر الي اسمها pickup_datatime ودي فيتشر كان فيها حتى pickup_datetime نفسه لو كان داخل كـ timestamp (رقم كبير):
# الموديل كان قادر يحفظ patterns زمنية شبه مباشرة
# ده نوع من implicit overfitting

# First try
# train RMSE = 0.5550 - R2 = 0.1353
# test RMSE = 0.6476 - R2 = 0.0672

# Second try
# train RMSE = 0.1411 - R2 = 0.7801
# test RMSE = 0.6499 - R2 = 0.0638

# After handling Overfitting
# train RMSE = 0.3812 - R2 = 0.4061
# test RMSE = 0.4105 - R2 = 0.4087

# train RMSE = 0.3736 - R2 = 0.4179
# test RMSE = 0.4127 - R2 = 0.4055


#train RMSE = 0.3889 - R2 = 0.3941
# test RMSE = 0.3978 - R2 = 0.4271
# ليه Train نزل بس Test طلع؟ بعد ما شيلت ال correlation between dayofyear and month features
# تشبيه بسيط:
# كنت حافظ الامتحان → درجات تدريب عالية
# بدأت تفهم المنهج → درجات امتحان أعلى


#We have reached the end of this kernel.
# We analysed 'Sale Price' and all the numerical and categorical features.
# We also studied discrete and continuous variables in our dataset separately.
# Studied Datetime features .
# We dealt with missing values , correlation among the features and target variable.
# We performed log transformation for skewed variables.
# Then Feature Scaling,Feature Selection
# And finally build three models using linear regression,Gradient Boosting and XGBoost Regressor.
# Linear regression gave the least accuracy of 87%.
# Gradient Boosting after hyper tuning the parameters gave a better accuracy of 98%.
# XGBoost Regressor gave the best accuracy with 99%.