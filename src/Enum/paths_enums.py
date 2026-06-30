from enum import Enum


class PathEnum(str, Enum):
    TRAIN_PATH = r"D:\AI_Track\ML_Course_Mustafa_Saad\Projects\Trip_Duration\src\data\train.csv"
    VAL_PATH = r"D:\AI_Track\ML_Course_Mustafa_Saad\Projects\Trip_Duration\src\data\val.csv"
    TRAIN_VAL_PATH = r"D:\AI_Track\ML_Course_Mustafa_Saad\Projects\Trip_Duration\src\data\train_val.csv"
    TEST_PATH = r"D:\AI_Track\ML_Course_Mustafa_Saad\Projects\Trip_Duration\src\data\test.csv"