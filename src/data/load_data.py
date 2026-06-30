"""
Module Documentation:
    load dataframe
    load x and t
    split train and val
    split x_train, x_val, t_train, t_val
"""

from sklearn.model_selection import train_test_split
import pandas as pd
from Config.load import load_config
config = load_config()


def load_df(path):
    if path is None:
        raise ValueError('Path is don`t defined correctly')
    df = pd.read_csv(path)
    return df


def load_x_t(df: pd.DataFrame):
    target = "trip_duration"
    x = df.drop(columns=[target])
    t = df[target]
    return x,t


def split_train_val(df: pd.DataFrame, split_sz=0.8):
    sz = int(split_sz*df.shape[0])
    train, val = df.iloc[:sz,:], df.iloc[sz:,:]
    return train, val


def split_data(x,t, split_sz=0.2):
    x_train, x_val, t_train, t_val = train_test_split(x, t, test_size=split_sz, random_state=config['RANDOM_STATE'])
    return x_train, x_val, t_train, t_val