
# Importing packages
import pandas as pd
import numpy as np
import pickle
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler



def clean_data(X, y, test=True):
    if not test:
        # Create scalers
        std_scaler = StandardScaler()
        mm_scaler = MinMaxScaler()
    else:
        # Load scalers that were fitted on training data
        with open("models/std_scaler.sav", "rb") as infile:
            std_scaler = pickle.load(infile)
        with open("models/mm_scaler.sav", "rb") as infile:
            mm_scaler = pickle.load(infile)
    
    df = pd.concat([X, y], axis=1)
    
    # Drop rows where target value is missing
    df.dropna(subset=["TARGET"], inplace=True)
    
    # Drop rows where important monetary values are missing
    df.dropna(subset=["AMT_INCOME_TOTAL", "AMT_CREDIT",
                      "AMT_ANNUITY", "AMT_GOODS_PRICE"],
              inplace=True)
    
    # Drop outliers
    if not test:
        idx = np.all(stats.zscore(df[["AMT_INCOME_TOTAL", "AMT_CREDIT",
                                      "AMT_ANNUITY", "AMT_GOODS_PRICE"]]) < 3, axis=1)
        df = df[idx]
    
    # Drop outlier from "DAYS_EMPLOYED" and the "SOCIAL_CIRCLE" columns
    df.drop(index=df[df["DAYS_EMPLOYED"] >= 50000].index, inplace=True)
    df.drop(index=df[df["OBS_30_CNT_SOCIAL_CIRCLE"] >= 100].index, inplace=True)
    df.drop(index=df[df["DEF_30_CNT_SOCIAL_CIRCLE"] >= 100].index, inplace=True)
    df.drop(index=df[df["OBS_60_CNT_SOCIAL_CIRCLE"] >= 100].index, inplace=True)
    df.drop(index=df[df["DEF_60_CNT_SOCIAL_CIRCLE"] >= 100].index, inplace=True)
    
    # Create list y with target values
    y = df["TARGET"].astype("int").astype("category")
    
    # Create DataFrame X for all features
    X = pd.DataFrame()
    
    # Copy already correct columns
    X["REGION_POPULATION_RELATIVE"] = df["REGION_POPULATION_RELATIVE"]
    
    # Convert data types
    X["FLAG_OWN_CAR"] = df["FLAG_OWN_CAR"].replace(["Y", "N"], [1, 0]).astype("int")
    X["FLAG_OWN_REALTY"] = df["FLAG_OWN_REALTY"].replace(["Y", "N"], [1, 0]).astype("int")
    X["REG_REGION_NOT_LIVE_REGION"] = df["REG_REGION_NOT_LIVE_REGION"].astype("int")
    X["REG_REGION_NOT_WORK_REGION"] = df["REG_REGION_NOT_WORK_REGION"].astype("int")
    X["LIVE_REGION_NOT_WORK_REGION"] = df["LIVE_REGION_NOT_WORK_REGION"].astype("int")
    X["REG_CITY_NOT_LIVE_CITY"] = df["REG_CITY_NOT_LIVE_CITY"].astype("int")
    X["REG_CITY_NOT_WORK_CITY"] = df["REG_CITY_NOT_WORK_CITY"].astype("int")
    X["FLAG_DOCUMENT_2"] = df["FLAG_DOCUMENT_2"].astype("int")
    X["FLAG_DOCUMENT_3"] = df["FLAG_DOCUMENT_3"].astype("int")
    X["FLAG_DOCUMENT_4"] = df["FLAG_DOCUMENT_4"].astype("int")
    X["FLAG_DOCUMENT_5"] = df["FLAG_DOCUMENT_5"].astype("int")
    X["FLAG_DOCUMENT_6"] = df["FLAG_DOCUMENT_6"].astype("int")
    X["FLAG_DOCUMENT_7"] = df["FLAG_DOCUMENT_7"].astype("int")
    X["FLAG_DOCUMENT_8"] = df["FLAG_DOCUMENT_8"].astype("int")
    X["FLAG_DOCUMENT_9"] = df["FLAG_DOCUMENT_9"].astype("int")
    X["FLAG_DOCUMENT_10"] = df["FLAG_DOCUMENT_10"].astype("int")
    X["FLAG_DOCUMENT_11"] = df["FLAG_DOCUMENT_11"].astype("int")
    X["FLAG_DOCUMENT_12"] = df["FLAG_DOCUMENT_12"].astype("int")
    X["FLAG_DOCUMENT_13"] = df["FLAG_DOCUMENT_13"].astype("int")
    X["FLAG_DOCUMENT_14"] = df["FLAG_DOCUMENT_14"].astype("int")
    X["FLAG_DOCUMENT_15"] = df["FLAG_DOCUMENT_15"].astype("int")
    X["FLAG_DOCUMENT_16"] = df["FLAG_DOCUMENT_16"].astype("int")
    X["FLAG_DOCUMENT_17"] = df["FLAG_DOCUMENT_17"].astype("int")
    X["FLAG_DOCUMENT_18"] = df["FLAG_DOCUMENT_18"].astype("int")
    X["FLAG_DOCUMENT_19"] = df["FLAG_DOCUMENT_19"].astype("int")
    X["FLAG_DOCUMENT_20"] = df["FLAG_DOCUMENT_20"].astype("int")
    X["FLAG_DOCUMENT_21"] = df["FLAG_DOCUMENT_21"].astype("int")
    
    # Create dummy variables for categorical columns
    X = pd.concat([X, pd.get_dummies(df[["NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_TYPE_SUITE",
                                         "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
                                         "NAME_HOUSING_TYPE", "ORGANIZATION_TYPE"]],
                                     drop_first=True)],  axis=1)
    
    # Filling all NaNs with mean values
    col_names = df.loc[:, "EXT_SOURCE_1" : "NONLIVINGAREA_MEDI"].columns
    X[col_names] = df[col_names].fillna(value=df[col_names].median())
    X["TOTALAREA_MODE"] = df["TOTALAREA_MODE"].fillna(value=df["TOTALAREA_MODE"].median())
    
    social_circle = ["OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
                     "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE"]
    X[social_circle] = df[social_circle].fillna(value=df[social_circle].median())
    
    enquiries = ["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY",
                 "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON",
                 "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR"]
    X[enquiries] = df[enquiries].fillna(value=df[enquiries].median())    
    
    # Making time span variables positive
    timespanes = ["DAYS_BIRTH", "DAYS_EMPLOYED",
                 "DAYS_REGISTRATION", "DAYS_ID_PUBLISH"]
    X[timespanes] = df[timespanes] * -1
    
    # Use Scaler
    std_scaled = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
                  "AMT_GOODS_PRICE", "REGION_RATING_CLIENT_W_CITY"]
    X["CNT_CHILDREN"] = df["CNT_CHILDREN"]
    mm_scaled = ["CNT_CHILDREN", "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION",
                 "DAYS_ID_PUBLISH", "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
                 "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE", "AMT_REQ_CREDIT_BUREAU_HOUR",
                 "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON",
                 "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR"]
        
    if test:
        X[std_scaled] = std_scaler.transform(df[std_scaled])
        X[mm_scaled] = mm_scaler.transform(X[mm_scaled])
    else:
        X[std_scaled] = std_scaler.fit_transform(df[std_scaled])
        X[mm_scaled] = mm_scaler.fit_transform(X[mm_scaled])
        # Write scalers into file
        pickle.dump(std_scaler, open("models/std_scaler.sav", "wb"))
        pickle.dump(mm_scaler, open("models/mm_scaler.sav", "wb"))

    return X, y