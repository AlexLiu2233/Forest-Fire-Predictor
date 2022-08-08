import datetime, re, math
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from numpy import std
import sklearn, sklearn.tree, sklearn.ensemble, sklearn.feature_extraction, sklearn.metrics
import pickle


def cleanup_data(df):
  #if it is a string (aka. 'N/A'), fill with nan. See part 1 for details
  def to_nan(entry):
    if type(entry) is not str: return entry
    else: return np.nan
  
  #fixing the missing data

  for column in df.columns:
    df[column] = df[column].apply(to_nan)  
    avg = df[column].mean()
    df[column].fillna(avg, inplace=True)    

   #fixing the missing data, end
   

  df['SIZE_HA'] = df['SIZE_HA'].values + 1 #OK

  df.drop(columns=['Unnamed: 18'], inplace = True)
  df.drop(columns=['Unnamed: 19'], inplace = True)

  #eliminating outliers
  q25, q75 = np.percentile(df['SIZE_HA'], 25), np.percentile(df['SIZE_HA'], 75)
  iqr = q75 - q25
  cut_off = iqr * 1.5
  lower, upper = q25 - cut_off, q75 + cut_off
  df= df[df['SIZE_HA'] <= upper]
  #eliminating outliers, end


  return df

df = pd.read_csv('test_data_milestone3.csv', low_memory=False)
df = cleanup_data(df)

#Split dataset to train and test sets
df_train = df[df['IGN_DATE'] < 20150000]
df_test = df[df['IGN_DATE'] >= 20150000]

#Split both train and test datasets into inputs and outputs
sizeHA_train = np.log(df_train['SIZE_HA'].values)
sizeHA_test = np.log(df_test['SIZE_HA'].values)
feats_train = df_train.drop(columns=['SIZE_HA'])
feats_test = df_test.drop(columns=['SIZE_HA'])

#Getting features
dvr = sklearn.feature_extraction.DictVectorizer()
enc_feats_train = dvr.fit_transform(feats_train.T.to_dict().values())
enc_feats_test = dvr.transform(feats_test.T.to_dict().values())

#Fitting random forest part of model
rf = sklearn.ensemble.RandomForestRegressor(n_estimators=80, max_samples=3406, max_features=0.7, min_samples_leaf=4, random_state=291, n_jobs=-1)
rf.fit(enc_feats_train, sizeHA_train)

pickle.dump(rf, open('random_forest.pk1', 'wb'))