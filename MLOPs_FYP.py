#!/usr/bin/env python
# coding: utf-8

# In[80]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pickle
from numpy import cumsum
from scipy.signal import savgol_filter
from scipy import stats
import plotly.graph_objects as go
from plotly.offline import iplot
from toolz import partition
from collections import defaultdict
import random
from scipy import ndimage
import random
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator


# In[64]:


import warnings
warnings.filterwarnings("ignore")


# In[65]:


def normal(data, nbr):
    hd = data.copy()
    h = hd.groupby([hd.index])['L_QUANTITY'].sum()
    l = len(h)
    sm = h.ewm(alpha=0.0005).mean()
    testing = pd.DataFrame(sm.values, columns=['L_QUANTITY'])
    testing.index = h.index
    testing = testing.asfreq('W')
    testing = testing.replace(np.nan, np.mean(testing['L_QUANTITY']))
    return testing


# In[66]:


def dateManipulation(testing, nbr, month):
    sigma = 2
    y_g1d = ndimage.gaussian_filter1d(testing['L_QUANTITY'].values, sigma)
    temp = testing.copy()
    temp.index = testing.index
    temp['ID'] = nbr
    temp['L_QUANTITY'] = y_g1d
    temp['tr_day'] = testing.index.day
    temp['tr_month'] = testing.index.month
    temp['tr_year'] = testing.index.year
    temp['stream_day'] = testing.index.day
    temp['stream_month'] = month[0]
    temp['stream_year'] = temp.index.year
    for i in range(1, len(month)):
        indices = temp[['stream_month']].sample(frac = 0.5, replace=True).index
        temp.loc[indices, 'stream_month'] = month[i]
    return temp


# In[67]:


def peakInjection(data, nbr, month):
    testing = data.copy()
    mean = testing['L_QUANTITY'].mean()
    limit1 = 0.50*mean
    for i in month:
        testing.loc[testing.index.month == i, 'L_QUANTITY'] += limit1
    temp = dateManipulation(testing, nbr, month)
    return temp


# In[72]:


def dtregressor():
    training1 = pd.read_csv('./mlops_training.csv')
    training1.dropna(inplace=True)
    training1.reset_index(drop=True, inplace=True)
    X = training1.drop(columns=['L_QUANTITY', 'stream_day', 'stream_month', 'stream_year', 'tr_year'])
    y = training1['L_QUANTITY']
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Model_Evaluation")
    max_depths = [10, 20, 30]
    for depth in max_depths:
        name = 'depth_'+str(depth) 
        with mlflow.start_run(run_name = name, description = 'Model Evaluation Runs'):
            regr = DecisionTreeRegressor(max_depth = depth)
            regr.fit(train_x, train_y)
            #pickle.dump(regr, open('./mlops_models/DTree.pkl', 'wb'))
            pred = regr.predict(test_x)
            sigma = 2
            pred1 = ndimage.gaussian_filter1d(pred, sigma)
            score = mean_squared_error(test_y, pred1)
            mlflow.log_metric('MSE', score)
    return pred1


# In[73]:


def preprocessing_and_training():
    df = pd.read_csv('./mlops_stream.csv')
    df['O_ORDERDATE'] = df['O_ORDERDATE'].astype('datetime64[ns]')
    df = df.set_index('O_ORDERDATE', drop=False)
    training = pd.DataFrame()
    count = 0
    randM = 1
    ids = df['L_PARTKEY'].unique()
    size = len(ids)
    parts = np.array_split(ids, 4)
    for part in parts:
        dfs_to_concat = []
        for i in part:
            tempD = normal(df[df['L_PARTKEY'] == i], i)
            peaked = peakInjection(tempD, i, [randM])
            dfs_to_concat.append(peaked)
        training = pd.concat(dfs_to_concat)  # Change append to concat
        training.to_csv('./mlops_training.csv', index=False)
        dtregressor()
        randM += 1

# In[77]:


def model_selection_and_deployment():
    exp_name = 'Model_Evaluation'
    exp_id = mlflow.get_experiment_by_name(exp_name).experiment_id
    all_runs = mlflow.search_runs(experiment_ids=exp_id, order_by=['metrics.MSE'])
    best_run = all_runs.loc[all_runs['metrics.MSE'].idxmin()]
    best_run_id = best_run['run_id']
    path = f"runs:/{best_run_id}/model"
#     load_model = mlflow.sklearn.load_model(path)
    mlflow.register_model(model_uri=path, name="Production_Model")


# In[ ]:
# preprocessing_and_training()
# model_selection_and_deployment()

airflow_args = {
    'owner':'razi',
    'depends_on_past':False,
    'start_date': datetime(2023, 5, 5),
    'retries':1,
    'retry_delay':timedelta(minutes=2) ,
    'user':'admin',
    'password':'admin'
}

dag = DAG(
    'ML_Pipeline',
    default_args=airflow_args,
    description='End-to-end ML pipline for FYP',
    schedule=timedelta(days=1)
)

with dag:
    preprocessing_training = PythonOperator(task_id='preprocess_data_and_training', python_callable=preprocessing_and_training)
    eval_and_selection = PythonOperator(task_id='select_the_best_model_for_deployment', python_callable=model_selection_and_deployment)

    #task order
    preprocessing_training >> eval_and_selection

