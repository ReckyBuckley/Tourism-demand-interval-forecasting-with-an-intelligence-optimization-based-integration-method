# -*- coding: utf-8 -*-

all_path="main"

loc_path=all_path+'\\code'
import os
os.chdir(loc_path)
from pandas import read_csv
import numpy as np
import pandas as pd
import pmdarima as pm
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import datetime


volume_path=all_path+'\\data\\Historical Tourist Data'
filename=volume_path+'\\Historical tourists of Jiuzhaigou.csv'
volume1=read_csv(filename,encoding='utf-8')
filename=volume_path+'\\Historical tourists of Mount Siguniang.csv'
volume2=read_csv(filename,encoding='utf-8')
review_path=all_path+'\\data\\Travel Website Reviews'
filename=review_path+'\\Travel Website Reviews for Jiuzhaigou.csv'
review1=read_csv(filename,encoding='UTF-8')
filename=review_path+'\\Travel Website Reviews for Mount Siguniang.csv'
review2=read_csv(filename,encoding='UTF-8')
weibo_path=all_path+'\\data\\Social Reviews'
filename=weibo_path+'\\Social Reviews for Jiuzhaigou.csv'
weibo1=read_csv(filename,encoding='UTF-8')
filename=weibo_path+'\\Social Reviews for Mount Siguniang.csv'
weibo2=read_csv(filename,encoding='UTF-8')
search_path=all_path+'\\data\\Search Engine'
filename=search_path+'\\Search Engine for Jiuzhaigou.csv'
search1=read_csv(filename,encoding='utf-8')
filename=search_path+'\\Search Engine for Mount Siguniang.csv'
search2=read_csv(filename,encoding='utf-8')

def shenzhan(sub_inputs,ahead):
    all_sequence=np.zeros([sub_inputs.shape[0]-ahead,ahead*sub_inputs.shape[1]])
    for i in range(sub_inputs.shape[0]-ahead):
        sub_matrix=sub_inputs[i:i+ahead]
        for j in range(sub_matrix.shape[0]):
            if j==0:
                sub_sequence=sub_matrix.values[j,:].reshape(-1,1).T
            if j>0:
                sub_sequence=np.concatenate((sub_sequence,sub_matrix.values[j,:].reshape(-1,1).T),axis = 1)
        all_sequence[i,:]=sub_sequence
    return all_sequence
def guiyihua(input_fit,input_test,y):
    input_scaler = MinMaxScaler()
    input_fit_n=np.zeros([input_fit.shape[0],input_fit.shape[1]])
    input_test_n=np.zeros([input_test.shape[1],1])
    for i in range(input_fit.shape[1]):
        input_scaler = input_scaler.fit(input_fit.values[:,i].reshape(-1, 1)) 
        input_fit_n[:,i] = np.squeeze(input_scaler.transform(input_fit.values[:,i].reshape(-1, 1)))
        input_test_n[i,:] = np.squeeze(input_scaler.transform(input_test.values[:,i].reshape(-1, 1)))
    input_test_n=input_test_n.T
    output_scaler=MinMaxScaler()
    y=output_scaler.fit_transform(y)
    return output_scaler,input_fit_n,input_test_n,y  

def calculate_mape(actual, forecast):
    actual, forecast = np.array(actual), np.array(forecast)
    return np.mean(np.abs((actual - forecast) / actual))
def calculate_rmse(actual, forecast):
    actual, forecast = np.array(actual), np.array(forecast)
    return (np.mean((actual - forecast)**2 ) )**0.5
def calculate_mae(actual, forecast):
    actual, forecast = np.array(actual), np.array(forecast)
    return np.mean(np.abs(actual - forecast)) 
def calculate_ia(actual, forecast):
    actual, forecast = np.array(actual), np.array(forecast)
    return 1-(np.mean((actual - forecast)**2))/(np.mean((np.abs(actual-np.mean(actual))+(np.abs(actual-np.mean(actual))))**2))

ahead=4
# Jiuzhaigou

matrix1=pd.DataFrame(np.empty((62, 3)))
matrix2=pd.DataFrame(np.empty((62, 3)))
matrix3=pd.DataFrame(np.empty((62, 3)))
matrix4=pd.DataFrame(np.empty((62, 3)))
matrix5=pd.DataFrame(np.empty((62, 3)))
matrix6=pd.DataFrame(np.empty((62, 3)))
matrix7=pd.DataFrame(np.empty((62, 3)))

arima_params = []

metrics_results = pd.DataFrame(columns=['Step', 'Forecast Model', 'RMSE','MAE','IA','MAPE'])

for step in [1,2,3]:

    # 1.volume+search
    fore_mode=1
    input_volume1=pd.concat([volume1],axis=1)   
    input_search=pd.concat([search1],axis=1)   
    input_volume1=shenzhan(input_volume1,ahead)
    input_search=shenzhan(input_search,ahead)
    arimax_start_t=datetime.datetime.now()
    arimax_models=[]
    arimax_fore=[]
    n_component_list=[]
    for i in range(62):
        y=volume1[ahead+step-1:(len(volume1)-62+i)]
        y=pd.DataFrame(y.values)
        actual=volume1[(len(volume1)-62):len(volume1)]
        actual=pd.DataFrame(actual.values)
        sub_input_volume1=input_volume1[0:(len(volume1)-62+i-step-ahead+1)]
        input_test_volume1=input_volume1[(len(volume1)-62+i-step-ahead+1):(len(volume1)-62+i-step-ahead+2)]
        sub_input_volume1=pd.DataFrame(sub_input_volume1)
        input_test_volume1=pd.DataFrame(input_test_volume1)
        sub_input_search=input_search[0:(len(volume1)-62+i-step-ahead+1)]
        input_test_search=input_search[(len(volume1)-62+i-step-ahead+1):(len(volume1)-62+i-step-ahead+2)]
        n_components_volume1=2
        volume1_pca = PCA(n_components=n_components_volume1)
        sub_input_volume1_pca = pd.DataFrame(volume1_pca.fit_transform(sub_input_volume1))
        input_test_volume1_pca = pd.DataFrame(volume1_pca.transform(input_test_volume1))
        while np.sum(volume1_pca.explained_variance_ratio_)<0.80:
            n_components_volume1=n_components_volume1+1
            volume1_pca = PCA(n_components=n_components_volume1)
            sub_input_volume1_pca = pd.DataFrame(volume1_pca.fit_transform(sub_input_volume1))
            input_test_volume1_pca = pd.DataFrame(volume1_pca.transform(input_test_volume1))

        n_components_search=2
        search_pca = PCA(n_components=n_components_search)
        sub_input_search_pca = pd.DataFrame(search_pca.fit_transform(sub_input_search))
        input_test_search_pca = pd.DataFrame(search_pca.transform(input_test_search))
        while np.sum(search_pca.explained_variance_ratio_)<0.80:
            n_components_search=n_components_search+1
            search_pca = PCA(n_components=n_components_search)
            sub_input_search_pca = pd.DataFrame(search_pca.fit_transform(sub_input_search))
            input_test_search_pca = pd.DataFrame(search_pca.transform(input_test_search))    
        sub_inputs_pca=pd.concat([sub_input_volume1,sub_input_search_pca],axis=1)
        input_test_pca=pd.concat([input_test_volume1,input_test_search_pca],axis=1)
        output_scaler,input_fit_n,input_test_n,y=guiyihua(sub_inputs_pca,input_test_pca,y)
        model = pm.auto_arima(y=y,X=input_fit_n,seasonal=True, information_criterion='aic')
        p, d, q = model.order
        arima_params.append((p, d, q))
        
        # make your forecasts
        ini_fore = model.predict(1,input_test_n)
        fin_fore=output_scaler.inverse_transform(ini_fore.reshape(-1,1))
        if fin_fore<0:
            fin_fore=0
        
        arimax_models.append(model)
        arimax_fore.append(float(fin_fore))
        sub_components=[n_components_search]
        n_component_list.append(sub_components)
    n_component_list=np.array(n_component_list)
    arimax_fore=np.array(arimax_fore)
    arimax_end_t=datetime.datetime.now()    
    arimax_run_time=((arimax_end_t-arimax_start_t).seconds)    
    matrix1[step-1]=arimax_fore
    rmse = calculate_rmse(actual, pd.DataFrame(arimax_fore))
    mae = calculate_mae(actual, pd.DataFrame(arimax_fore))
    ia = calculate_ia(actual, pd.DataFrame(arimax_fore))
    mape = calculate_mape(actual, pd.DataFrame(arimax_fore))  
        
    metrics_results = metrics_results.append({
        'Step': step,
        'Forecast Model': f'Model {fore_mode}',
        'RMSE':rmse,
        'MAE':mae,        
        'IA':ia,
        'MAPE': mape
    }, ignore_index=True)
        
    # 2.volume+review
    fore_mode=2
    input_volume1=pd.concat([volume1],axis=1)
    input_review=pd.concat([review1],axis=1)
    input_volume1=shenzhan(input_volume1,ahead)
    input_review=shenzhan(input_review,ahead)
    arimax_start_t=datetime.datetime.now()
    arimax_models=[]
    arimax_fore=[]
    n_component_list=[]
    for i in range(62):
        y=volume1[ahead+step-1:(len(volume1)-62+i)]
        y=pd.DataFrame(y.values)
        actual=volume1[(len(volume1)-62):len(volume1)]
        actual=pd.DataFrame(actual.values)
        sub_input_volume1=input_volume1[0:(len(volume1)-62+i-step-ahead+1)]
        input_test_volume1=input_volume1[(len(volume1)-62+i-step-ahead+1):(len(volume1)-62+i-step-ahead+2)]
        sub_input_volume1=pd.DataFrame(sub_input_volume1)
        input_test_volume1=pd.DataFrame(input_test_volume1)
        sub_input_review=input_review[0:(len(volume1)-62+i-step-ahead+1)]
        input_test_review=input_review[(len(volume1)-62+i-step-ahead+1):(len(volume1)-62+i-step-ahead+2)]
        n_components_volume1=2
        volume1_pca = PCA(n_components=n_components_volume1)
        sub_input_volume1_pca = pd.DataFrame(volume1_pca.fit_transform(sub_input_volume1))
        input_test_volume1_pca = pd.DataFrame(volume1_pca.transform(input_test_volume1))
        while np.sum(volume1_pca.explained_variance_ratio_)<0.80:
            n_components_volume1=n_components_volume1+1
            volume1_pca = PCA(n_components=n_components_volume1)
            sub_input_volume1_pca = pd.DataFrame(volume1_pca.fit_transform(sub_input_volume1))
            input_test_volume1_pca = pd.DataFrame(volume1_pca.transform(input_test_volume1))

        n_components_review=2
        review_pca = PCA(n_components=n_components_review)
        sub_input_review_pca = pd.DataFrame(review_pca.fit_transform(sub_input_review))
        input_test_review_pca = pd.DataFrame(review_pca.transform(input_test_review))
        while np.sum(review_pca.explained_variance_ratio_)<0.80:
            n_components_review=n_components_review+1
            review_pca = PCA(n_components=n_components_review)
            sub_input_review_pca = pd.DataFrame(review_pca.fit_transform(sub_input_review))
            input_test_review_pca = pd.DataFrame(review_pca.transform(input_test_review))    
        
        sub_inputs_pca=pd.concat([sub_input_volume1,sub_input_review_pca],axis=1)
        input_test_pca=pd.concat([input_test_volume1,input_test_review_pca],axis=1)
        output_scaler,input_fit_n,input_test_n,y=guiyihua(sub_inputs_pca,input_test_pca,y)
        model = pm.auto_arima(y=y,X=input_fit_n,seasonal=True, information_criterion='aic')
        p, d, q = model.order  
        arima_params.append((p, d, q))  
       
        # make your forecasts
        ini_fore = model.predict(1,input_test_n) 
        fin_fore=output_scaler.inverse_transform(ini_fore.reshape(-1,1))
        if fin_fore<0:
            fin_fore=0
        
        arimax_models.append(model)
        arimax_fore.append(float(fin_fore))
        sub_components=[n_components_review]
        n_component_list.append(sub_components)
    n_component_list=np.array(n_component_list)
    arimax_fore=np.array(arimax_fore)
    arimax_end_t=datetime.datetime.now()    
    arimax_run_time=((arimax_end_t-arimax_start_t).seconds)    
    matrix2[step-1]=arimax_fore
    rmse = calculate_rmse(actual, pd.DataFrame(arimax_fore))
    mae = calculate_mae(actual, pd.DataFrame(arimax_fore))
    ia = calculate_ia(actual, pd.DataFrame(arimax_fore))
    mape = calculate_mape(actual, pd.DataFrame(arimax_fore))  
        
    metrics_results = metrics_results.append({
        'Step': step,
        'Forecast Model': f'Model {fore_mode}',
        'RMSE':rmse,
        'MAE':mae,       
        'IA':ia,
        'MAPE': mape
    }, ignore_index=True)
    
    # 3.volume+weibo
    fore_mode=3
    input_volume1=pd.concat([volume1],axis=1)    
    input_weibo=pd.concat([weibo1],axis=1)
    
    input_volume1=shenzhan(input_volume1,ahead)
    input_weibo=shenzhan(input_weibo,ahead)
    arimax_start_t=datetime.datetime.now()
    arimax_models=[]
    arimax_fore=[]
    n_component_list=[]
    for i in range(62):
        y=volume1[ahead+step-1:(len(volume1)-62+i)]
        y=pd.DataFrame(y.values)
        actual=volume1[(len(volume1)-62):len(volume1)]
        actual=pd.DataFrame(actual.values)
        sub_input_volume1=input_volume1[0:(len(volume1)-62+i-step-ahead+1)]
        input_test_volume1=input_volume1[(len(volume1)-62+i-step-ahead+1):(len(volume1)-62+i-step-ahead+2)]
        sub_input_volume1=pd.DataFrame(sub_input_volume1)
        input_test_volume1=pd.DataFrame(input_test_volume1)
        sub_input_weibo=input_weibo[0:(len(volume1)-62+i-step-ahead+1)]
        input_test_weibo=input_weibo[(len(volume1)-62+i-step-ahead+1):(len(volume1)-62+i-step-ahead+2)]
        n_components_volume1=2
        volume1_pca = PCA(n_components=n_components_volume1)
        sub_input_volume1_pca = pd.DataFrame(volume1_pca.fit_transform(sub_input_volume1))
        input_test_volume1_pca = pd.DataFrame(volume1_pca.transform(input_test_volume1))
        while np.sum(volume1_pca.explained_variance_ratio_)<0.80:
            n_components_volume1=n_components_volume1+1
            volume1_pca = PCA(n_components=n_components_volume1)
            sub_input_volume1_pca = pd.DataFrame(volume1_pca.fit_transform(sub_input_volume1))
            input_test_volume1_pca = pd.DataFrame(volume1_pca.transform(input_test_volume1))
        n_components_weibo=2
        weibo_pca = PCA(n_components=n_components_weibo)
        sub_input_weibo_pca = pd.DataFrame(weibo_pca.fit_transform(sub_input_weibo))
        input_test_weibo_pca = pd.DataFrame(weibo_pca.transform(input_test_weibo))
        while np.sum(weibo_pca.explained_variance_ratio_)<0.80:
            n_components_weibo=n_components_weibo+1
            weibo_pca = PCA(n_components=n_components_weibo)
            sub_input_weibo_pca = pd.DataFrame(weibo_pca.fit_transform(sub_input_weibo))
            input_test_weibo_pca = pd.DataFrame(weibo_pca.transform(input_test_weibo))    
        sub_inputs_pca=pd.concat([sub_input_volume1,sub_input_weibo_pca],axis=1)
        input_test_pca=pd.concat([input_test_volume1,input_test_weibo_pca],axis=1)
        output_scaler,input_fit_n,input_test_n,y=guiyihua(sub_inputs_pca,input_test_pca,y)
        model = pm.auto_arima(y=y,X=input_fit_n,seasonal=True, information_criterion='aic')
        p, d, q = model.order  
        arima_params.append((p, d, q))  
        
        # make your forecasts
        ini_fore = model.predict(1,input_test_n) 
        fin_fore=output_scaler.inverse_transform(ini_fore.reshape(-1,1))
        if fin_fore<0:
            fin_fore=0
        
        arimax_models.append(model)
        arimax_fore.append(float(fin_fore))
        sub_components=[n_components_weibo]
        n_component_list.append(sub_components)
    n_component_list=np.array(n_component_list)
    arimax_fore=np.array(arimax_fore)
    arimax_end_t=datetime.datetime.now()    
    arimax_run_time=((arimax_end_t-arimax_start_t).seconds)    
    matrix3[step-1]=arimax_fore
    rmse = calculate_rmse(actual, pd.DataFrame(arimax_fore))
    mae = calculate_mae(actual, pd.DataFrame(arimax_fore))    
    ia = calculate_ia(actual, pd.DataFrame(arimax_fore))
    mape = calculate_mape(actual, pd.DataFrame(arimax_fore))  
        
    metrics_results = metrics_results.append({
        'Step': step,
        'Forecast Model': f'Model {fore_mode}',
        'RMSE':rmse,
        'MAE':mae,        
        'IA':ia,
        'MAPE': mape
    }, ignore_index=True)
    
    # 4.volume+search+weibo
    fore_mode=4
    input_volume1=pd.concat([volume1],axis=1)    
    input_search=pd.concat([search1],axis=1)
    input_weibo=pd.concat([weibo1],axis=1)
        
    input_volume1=shenzhan(input_volume1,ahead)
    input_search=shenzhan(input_search,ahead)
    input_weibo=shenzhan(input_weibo,ahead)
    arimax_start_t=datetime.datetime.now()
    arimax_models=[]
    arimax_fore=[]
    n_component_list=[]
    for i in range(62):
        y=volume1[ahead+step-1:(len(volume1)-62+i)]
        y=pd.DataFrame(y.values)
        actual=volume1[(len(volume1)-62):len(volume1)]
        actual=pd.DataFrame(actual.values)
        sub_input_volume1=input_volume1[0:(len(volume1)-62+i-step-ahead+1)]
        input_test_volume1=input_volume1[(len(volume1)-62+i-step-ahead+1):(len(volume1)-62+i-step-ahead+2)]
        sub_input_volume1=pd.DataFrame(sub_input_volume1)
        input_test_volume1=pd.DataFrame(input_test_volume1)
        sub_input_search=input_search[0:(len(volume1)-62+i-step-ahead+1)]
        input_test_search=input_search[(len(volume1)-62+i-step-ahead+1):(len(volume1)-62+i-step-ahead+2)]
        sub_input_weibo=input_weibo[0:(len(volume1)-62+i-step-ahead+1)]
        input_test_weibo=input_weibo[(len(volume1)-62+i-step-ahead+1):(len(volume1)-62+i-step-ahead+2)]
        n_components_volume1=2
        volume1_pca = PCA(n_components=n_components_volume1)
        sub_input_volume1_pca = pd.DataFrame(volume1_pca.fit_transform(sub_input_volume1))
        input_test_volume1_pca = pd.DataFrame(volume1_pca.transform(input_test_volume1))
        while np.sum(volume1_pca.explained_variance_ratio_)<0.80:
            n_components_volume1=n_components_volume1+1
            volume1_pca = PCA(n_components=n_components_volume1)
            sub_input_volume1_pca = pd.DataFrame(volume1_pca.fit_transform(sub_input_volume1))
            input_test_volume1_pca = pd.DataFrame(volume1_pca.transform(input_test_volume1))

        n_components_search=2
        search_pca = PCA(n_components=n_components_search)
        sub_input_search_pca = pd.DataFrame(search_pca.fit_transform(sub_input_search))
        input_test_search_pca = pd.DataFrame(search_pca.transform(input_test_search))
        while np.sum(search_pca.explained_variance_ratio_)<0.80:
            n_components_search=n_components_search+1
            search_pca = PCA(n_components=n_components_search)
            sub_input_search_pca = pd.DataFrame(search_pca.fit_transform(sub_input_search))
            input_test_search_pca = pd.DataFrame(search_pca.transform(input_test_search))
        n_components_weibo=2
        weibo_pca = PCA(n_components=n_components_weibo)
        sub_input_weibo_pca = pd.DataFrame(weibo_pca.fit_transform(sub_input_weibo))
        input_test_weibo_pca = pd.DataFrame(weibo_pca.transform(input_test_weibo))
        while np.sum(weibo_pca.explained_variance_ratio_)<0.80:
            n_components_weibo=n_components_weibo+1
            weibo_pca = PCA(n_components=n_components_weibo)
            sub_input_weibo_pca = pd.DataFrame(weibo_pca.fit_transform(sub_input_weibo))
            input_test_weibo_pca = pd.DataFrame(weibo_pca.transform(input_test_weibo))    
        sub_inputs_pca=pd.concat([sub_input_volume1,sub_input_search_pca,sub_input_weibo_pca],axis=1)
        input_test_pca=pd.concat([input_test_volume1,input_test_search_pca,input_test_weibo_pca],axis=1)
        output_scaler,input_fit_n,input_test_n,y=guiyihua(sub_inputs_pca,input_test_pca,y)
        model = pm.auto_arima(y=y,X=input_fit_n,seasonal=True, information_criterion='aic')
        p, d, q = model.order  
        arima_params.append((p, d, q))  
        
        # make your forecasts
        ini_fore = model.predict(1,input_test_n) 
        fin_fore=output_scaler.inverse_transform(ini_fore.reshape(-1,1))
        if fin_fore<0:
            fin_fore=0
        
        arimax_models.append(model)
        arimax_fore.append(float(fin_fore))
        sub_components=[n_components_search,n_components_weibo]
        n_component_list.append(sub_components)
    n_component_list=np.array(n_component_list)
    arimax_fore=np.array(arimax_fore)
    arimax_end_t=datetime.datetime.now()    
    arimax_run_time=((arimax_end_t-arimax_start_t).seconds)    
    matrix4[step-1]=arimax_fore
    rmse = calculate_rmse(actual, pd.DataFrame(arimax_fore))
    mae = calculate_mae(actual, pd.DataFrame(arimax_fore))
    ia = calculate_ia(actual, pd.DataFrame(arimax_fore))
    mape = calculate_mape(actual, pd.DataFrame(arimax_fore))  
        
    metrics_results = metrics_results.append({
        'Step': step,
        'Forecast Model': f'Model {fore_mode}',
        'RMSE':rmse,
        'MAE':mae,        
        'IA':ia,
        'MAPE': mape
    }, ignore_index=True)
    
    # 5.volume+review+weibo
    fore_mode=5
    input_volume1=pd.concat([volume1],axis=1)    
    input_review=pd.concat([review1],axis=1)
    input_weibo=pd.concat([weibo1],axis=1)
    
    input_volume1=shenzhan(input_volume1,ahead)
    input_review=shenzhan(input_review,ahead)
    input_weibo=shenzhan(input_weibo,ahead)
    arimax_start_t=datetime.datetime.now()
    arimax_models=[]
    arimax_fore=[]
    n_component_list=[]
    for i in range(62):
        y=volume1[ahead+step-1:(len(volume1)-62+i)]
        y=pd.DataFrame(y.values)
        actual=volume1[(len(volume1)-62):len(volume1)]
        actual=pd.DataFrame(actual.values)
        sub_input_volume1=input_volume1[0:(len(volume1)-62+i-step-ahead+1)]
        input_test_volume1=input_volume1[(len(volume1)-62+i-step-ahead+1):(len(volume1)-62+i-step-ahead+2)]
        sub_input_volume1=pd.DataFrame(sub_input_volume1)
        input_test_volume1=pd.DataFrame(input_test_volume1)
        sub_input_review=input_review[0:(len(volume1)-62+i-step-ahead+1)]
        input_test_review=input_review[(len(volume1)-62+i-step-ahead+1):(len(volume1)-62+i-step-ahead+2)]
        sub_input_weibo=input_weibo[0:(len(volume1)-62+i-step-ahead+1)]
        input_test_weibo=input_weibo[(len(volume1)-62+i-step-ahead+1):(len(volume1)-62+i-step-ahead+2)]
        n_components_volume1=2
        volume1_pca = PCA(n_components=n_components_volume1)
        sub_input_volume1_pca = pd.DataFrame(volume1_pca.fit_transform(sub_input_volume1))
        input_test_volume1_pca = pd.DataFrame(volume1_pca.transform(input_test_volume1))
        while np.sum(volume1_pca.explained_variance_ratio_)<0.80:
            n_components_volume1=n_components_volume1+1
            volume1_pca = PCA(n_components=n_components_volume1)
            sub_input_volume1_pca = pd.DataFrame(volume1_pca.fit_transform(sub_input_volume1))
            input_test_volume1_pca = pd.DataFrame(volume1_pca.transform(input_test_volume1))

        n_components_review=2
        review_pca = PCA(n_components=n_components_review)
        sub_input_review_pca = pd.DataFrame(review_pca.fit_transform(sub_input_review))
        input_test_review_pca = pd.DataFrame(review_pca.transform(input_test_review))
        while np.sum(review_pca.explained_variance_ratio_)<0.80:
            n_components_review=n_components_review+1
            review_pca = PCA(n_components=n_components_review)
            sub_input_review_pca = pd.DataFrame(review_pca.fit_transform(sub_input_review))
            input_test_review_pca = pd.DataFrame(review_pca.transform(input_test_review))
        n_components_weibo=2
        weibo_pca = PCA(n_components=n_components_weibo)
        sub_input_weibo_pca = pd.DataFrame(weibo_pca.fit_transform(sub_input_weibo))
        input_test_weibo_pca = pd.DataFrame(weibo_pca.transform(input_test_weibo))
        while np.sum(weibo_pca.explained_variance_ratio_)<0.80:
            n_components_weibo=n_components_weibo+1
            weibo_pca = PCA(n_components=n_components_weibo)
            sub_input_weibo_pca = pd.DataFrame(weibo_pca.fit_transform(sub_input_weibo))
            input_test_weibo_pca = pd.DataFrame(weibo_pca.transform(input_test_weibo))
        sub_inputs_pca=pd.concat([sub_input_volume1,sub_input_review_pca,sub_input_weibo_pca],axis=1)
        input_test_pca=pd.concat([input_test_volume1,input_test_review_pca,input_test_weibo_pca],axis=1)
        output_scaler,input_fit_n,input_test_n,y=guiyihua(sub_inputs_pca,input_test_pca,y)
        model = pm.auto_arima(y=y,X=input_fit_n,seasonal=True, information_criterion='aic')
        p, d, q = model.order  
        arima_params.append((p, d, q))  
        
        # make your forecasts
        ini_fore = model.predict(1,input_test_n) 
        fin_fore=output_scaler.inverse_transform(ini_fore.reshape(-1,1))
        if fin_fore<0:
            fin_fore=0
        
        arimax_models.append(model)
        arimax_fore.append(float(fin_fore))
        sub_components=[n_components_review,n_components_weibo]
        n_component_list.append(sub_components)
    n_component_list=np.array(n_component_list)
    arimax_fore=np.array(arimax_fore)
    arimax_end_t=datetime.datetime.now()    
    arimax_run_time=((arimax_end_t-arimax_start_t).seconds)
    matrix5[step-1]=arimax_fore
    rmse = calculate_rmse(actual, pd.DataFrame(arimax_fore))
    mae = calculate_mae(actual, pd.DataFrame(arimax_fore))    
    ia = calculate_ia(actual, pd.DataFrame(arimax_fore))
    mape = calculate_mape(actual, pd.DataFrame(arimax_fore))  
        
    metrics_results = metrics_results.append({
        'Step': step,
        'Forecast Model': f'Model {fore_mode}',
        'RMSE':rmse,
        'MAE':mae,        
        'IA':ia,
        'MAPE': mape
    }, ignore_index=True)
    
    # 6.volume+review+search
    fore_mode=6
    input_volume1=pd.concat([volume1],axis=1)    
    input_review=pd.concat([review1],axis=1) 
    input_search=pd.concat([search1],axis=1)
        
    input_volume1=shenzhan(input_volume1,ahead)
    input_review=shenzhan(input_review,ahead)
    input_search=shenzhan(input_search,ahead)
    arimax_start_t=datetime.datetime.now()
    arimax_models=[]
    arimax_fore=[]
    n_component_list=[]
    for i in range(62):
        y=volume1[ahead+step-1:(len(volume1)-62+i)]
        y=pd.DataFrame(y.values)
        actual=volume1[(len(volume1)-62):len(volume1)]
        actual=pd.DataFrame(actual.values)
        sub_input_volume1=input_volume1[0:(len(volume1)-62+i-step-ahead+1)]
        input_test_volume1=input_volume1[(len(volume1)-62+i-step-ahead+1):(len(volume1)-62+i-step-ahead+2)]
        sub_input_volume1=pd.DataFrame(sub_input_volume1)
        input_test_volume1=pd.DataFrame(input_test_volume1)
        sub_input_review=input_review[0:(len(volume1)-62+i-step-ahead+1)]
        input_test_review=input_review[(len(volume1)-62+i-step-ahead+1):(len(volume1)-62+i-step-ahead+2)]
        sub_input_search=input_search[0:(len(volume1)-62+i-step-ahead+1)]
        input_test_search=input_search[(len(volume1)-62+i-step-ahead+1):(len(volume1)-62+i-step-ahead+2)]
        
        n_components_volume1=2
        volume1_pca = PCA(n_components=n_components_volume1)
        sub_input_volume1_pca = pd.DataFrame(volume1_pca.fit_transform(sub_input_volume1))
        input_test_volume1_pca = pd.DataFrame(volume1_pca.transform(input_test_volume1))
        while np.sum(volume1_pca.explained_variance_ratio_)<0.80:
            n_components_volume1=n_components_volume1+1
            volume1_pca = PCA(n_components=n_components_volume1)
            sub_input_volume1_pca = pd.DataFrame(volume1_pca.fit_transform(sub_input_volume1))
            input_test_volume1_pca = pd.DataFrame(volume1_pca.transform(input_test_volume1))

        n_components_review=2
        review_pca = PCA(n_components=n_components_review)
        sub_input_review_pca = pd.DataFrame(review_pca.fit_transform(sub_input_review))
        input_test_review_pca = pd.DataFrame(review_pca.transform(input_test_review))
        while np.sum(review_pca.explained_variance_ratio_)<0.80:
            n_components_review=n_components_review+1
            review_pca = PCA(n_components=n_components_review)
            sub_input_review_pca = pd.DataFrame(review_pca.fit_transform(sub_input_review))
            input_test_review_pca = pd.DataFrame(review_pca.transform(input_test_review))
        
        n_components_search=2
        search_pca = PCA(n_components=n_components_search)
        sub_input_search_pca = pd.DataFrame(search_pca.fit_transform(sub_input_search))
        input_test_search_pca = pd.DataFrame(search_pca.transform(input_test_search))
        while np.sum(search_pca.explained_variance_ratio_)<0.80:
            n_components_search=n_components_search+1
            search_pca = PCA(n_components=n_components_search)
            sub_input_search_pca = pd.DataFrame(search_pca.fit_transform(sub_input_search))
            input_test_search_pca = pd.DataFrame(search_pca.transform(input_test_search))
        sub_inputs_pca=pd.concat([sub_input_volume1,sub_input_review_pca,sub_input_search_pca],axis=1)
        input_test_pca=pd.concat([input_test_volume1,input_test_review_pca,input_test_search_pca],axis=1)
        output_scaler,input_fit_n,input_test_n,y=guiyihua(sub_inputs_pca,input_test_pca,y)
        model = pm.auto_arima(y=y,X=input_fit_n,seasonal=True, information_criterion='aic')
        p, d, q = model.order  
        arima_params.append((p, d, q))  
        
        # make your forecasts
        ini_fore = model.predict(1,input_test_n) 
        fin_fore=output_scaler.inverse_transform(ini_fore.reshape(-1,1))
        if fin_fore<0:
            fin_fore=0
        
        arimax_models.append(model)
        arimax_fore.append(float(fin_fore))
        sub_components=[n_components_review,n_components_search]
        n_component_list.append(sub_components)
    n_component_list=np.array(n_component_list)
    arimax_fore=np.array(arimax_fore)
    arimax_end_t=datetime.datetime.now()    
    arimax_run_time=((arimax_end_t-arimax_start_t).seconds)    
    matrix6[step-1]=arimax_fore
    rmse = calculate_rmse(actual, pd.DataFrame(arimax_fore))
    mae = calculate_mae(actual, pd.DataFrame(arimax_fore)) 
    ia = calculate_ia(actual, pd.DataFrame(arimax_fore))
    mape = calculate_mape(actual, pd.DataFrame(arimax_fore))  
        
    metrics_results = metrics_results.append({
        'Step': step,
        'Forecast Model': f'Model {fore_mode}',
        'RMSE':rmse,
        'MAE':mae,        
        'IA':ia,
        'MAPE': mape
    }, ignore_index=True)
    
    # 7.volume+search+review+weibo
    fore_mode=7
    input_volume1=pd.concat([volume1],axis=1)
    
    input_search=pd.concat([search1],axis=1)    
    input_review=pd.concat([review1.iloc],axis=1)
    input_weibo=pd.concat([weibo1],axis=1)
       
    input_volume1=shenzhan(input_volume1,ahead)
    input_search=shenzhan(input_search,ahead)
    input_review=shenzhan(input_review,ahead)
    input_weibo=shenzhan(input_weibo,ahead)
    arimax_start_t=datetime.datetime.now()
    arimax_models=[]
    arimax_fore=[]
    n_component_list=[]
    for i in range(62):
        y=volume1[ahead+step-1:(len(volume1)-62+i)]
        y=pd.DataFrame(y.values)
        actual=volume1[(len(volume1)-62):len(volume1)]
        actual=pd.DataFrame(actual.values)
        sub_input_volume1=input_volume1[0:(len(volume1)-62+i-step-ahead+1)]
        input_test_volume1=input_volume1[(len(volume1)-62+i-step-ahead+1):(len(volume1)-62+i-step-ahead+2)]
        sub_input_volume1=pd.DataFrame(sub_input_volume1)
        input_test_volume1=pd.DataFrame(input_test_volume1)
        
        sub_input_search=input_search[0:(len(volume1)-62+i-step-ahead+1)]
        input_test_search=input_search[(len(volume1)-62+i-step-ahead+1):(len(volume1)-62+i-step-ahead+2)]
        sub_input_review=input_review[0:(len(volume1)-62+i-step-ahead+1)]
        input_test_review=input_review[(len(volume1)-62+i-step-ahead+1):(len(volume1)-62+i-step-ahead+2)]
        sub_input_weibo=input_weibo[0:(len(volume1)-62+i-step-ahead+1)]
        input_test_weibo=input_weibo[(len(volume1)-62+i-step-ahead+1):(len(volume1)-62+i-step-ahead+2)]
        
        n_components_search=2
        search_pca = PCA(n_components=n_components_search)
        sub_input_search_pca = pd.DataFrame(search_pca.fit_transform(sub_input_search))
        input_test_search_pca = pd.DataFrame(search_pca.transform(input_test_search))
        while np.sum(search_pca.explained_variance_ratio_)<0.80:
            n_components_search=n_components_search+1
            search_pca = PCA(n_components=n_components_search)
            sub_input_search_pca = pd.DataFrame(search_pca.fit_transform(sub_input_search))
            input_test_search_pca = pd.DataFrame(search_pca.transform(input_test_search))
        n_components_review=2
        review_pca = PCA(n_components=n_components_review)
        sub_input_review_pca = pd.DataFrame(review_pca.fit_transform(sub_input_review))
        input_test_review_pca = pd.DataFrame(review_pca.transform(input_test_review))
        while np.sum(review_pca.explained_variance_ratio_)<0.80:
            n_components_review=n_components_review+1
            review_pca = PCA(n_components=n_components_review)
            sub_input_review_pca = pd.DataFrame(review_pca.fit_transform(sub_input_review))
            input_test_review_pca = pd.DataFrame(review_pca.transform(input_test_review))
        n_components_weibo=2
        weibo_pca = PCA(n_components=n_components_weibo)
        sub_input_weibo_pca = pd.DataFrame(weibo_pca.fit_transform(sub_input_weibo))
        input_test_weibo_pca = pd.DataFrame(weibo_pca.transform(input_test_weibo))
        while np.sum(weibo_pca.explained_variance_ratio_)<0.80:
            n_components_weibo=n_components_weibo+1
            weibo_pca = PCA(n_components=n_components_weibo)
            sub_input_weibo_pca = pd.DataFrame(weibo_pca.fit_transform(sub_input_weibo))
            input_test_weibo_pca = pd.DataFrame(weibo_pca.transform(input_test_weibo))
        sub_inputs_pca=pd.concat([sub_input_search_pca,sub_input_review_pca,sub_input_weibo_pca],axis=1)
        input_test_pca=pd.concat([input_test_search_pca,input_test_review_pca,input_test_weibo_pca],axis=1)
        output_scaler,input_fit_n,input_test_n,y=guiyihua(sub_inputs_pca,input_test_pca,y)
        model = pm.auto_arima(y=y,X=input_fit_n,seasonal=True, information_criterion='aic')
        p, d, q = model.order  
        arima_params.append((p, d, q))  
        # make your forecasts
        ini_fore = model.predict(1,input_test_n) 
        fin_fore=output_scaler.inverse_transform(ini_fore.reshape(-1,1))
        if fin_fore<0:
            fin_fore=0
        
        arimax_models.append(model)
        arimax_fore.append(float(fin_fore))
        sub_components=[n_components_search,n_components_review,n_components_weibo]
        n_component_list.append(sub_components)
    n_component_list=np.array(n_component_list)
    arimax_fore=np.array(arimax_fore)
    arimax_end_t=datetime.datetime.now()    
    arimax_run_time=((arimax_end_t-arimax_start_t).seconds)    
    matrix7[step-1]=arimax_fore
    rmse = calculate_rmse(actual, pd.DataFrame(arimax_fore))
    mae = calculate_mae(actual, pd.DataFrame(arimax_fore))    
    ia = calculate_ia(actual, pd.DataFrame(arimax_fore))
    mape = calculate_mape(actual, pd.DataFrame(arimax_fore))  
        
    metrics_results = metrics_results.append({
        'Step': step,
        'Forecast Model': f'Model {fore_mode}',
        'RMSE':rmse,
        'MAE':mae,       
        'IA':ia,
        'MAPE': mape
    }, ignore_index=True)
    
df_params = pd.DataFrame(arima_params, columns=['p', 'd', 'q'])
   
matrix21=np.array([matrix1[0],matrix2[0],matrix3[0],matrix4[0],matrix5[0],matrix6[0],matrix7[0]])
matrix22=np.array([matrix1[1],matrix2[1],matrix3[1],matrix4[1],matrix5[1],matrix6[1],matrix7[1]])
matrix23=np.array([matrix1[2],matrix2[2],matrix3[2],matrix4[2],matrix5[2],matrix6[2],matrix7[2]])

matrix21=matrix21.T
matrix22=matrix22.T
matrix23=matrix23.T


# Mount Siguniang

matrix1=pd.DataFrame(np.empty((62, 3)))
matrix2=pd.DataFrame(np.empty((62, 3)))
matrix3=pd.DataFrame(np.empty((62, 3)))
matrix4=pd.DataFrame(np.empty((62, 3)))
matrix5=pd.DataFrame(np.empty((62, 3)))
matrix6=pd.DataFrame(np.empty((62, 3)))
matrix7=pd.DataFrame(np.empty((62, 3)))
arima_params = []


for step in [1,2,3]:

    # 1.volume+search
    fore_mode=1
    input_volume2=pd.concat([volume2],axis=1)   
    input_search=pd.concat([search2],axis=1)
        
    input_volume2=shenzhan(input_volume2,ahead)
    input_search=shenzhan(input_search,ahead)
    arimax_start_t=datetime.datetime.now()
    arimax_models=[]
    arimax_fore=[]
    n_component_list=[]
    for i in range(62):
        y=volume2[ahead+step-1:(len(volume2)-62+i)]
        y=pd.DataFrame(y.values)
        actual=volume2[(len(volume2)-62):len(volume2)]
        actual=pd.DataFrame(actual.values)
        sub_input_volume2=input_volume2[0:(len(volume2)-62+i-step-ahead+1)]
        input_test_volume2=input_volume2[(len(volume2)-62+i-step-ahead+1):(len(volume2)-62+i-step-ahead+2)]
        sub_input_volume2=pd.DataFrame(sub_input_volume2)
        input_test_volume2=pd.DataFrame(input_test_volume2)
        sub_input_search=input_search[0:(len(volume2)-62+i-step-ahead+1)]
        input_test_search=input_search[(len(volume2)-62+i-step-ahead+1):(len(volume2)-62+i-step-ahead+2)]
        n_components_volume2=2
        volume2_pca = PCA(n_components=n_components_volume2)
        sub_input_volume2_pca = pd.DataFrame(volume2_pca.fit_transform(sub_input_volume2))
        input_test_volume2_pca = pd.DataFrame(volume2_pca.transform(input_test_volume2))
        while np.sum(volume2_pca.explained_variance_ratio_)<0.80:
            n_components_volume2=n_components_volume2+1
            volume2_pca = PCA(n_components=n_components_volume2)
            sub_input_volume2_pca = pd.DataFrame(volume2_pca.fit_transform(sub_input_volume2))
            input_test_volume2_pca = pd.DataFrame(volume2_pca.transform(input_test_volume2))

        n_components_search=2
        search_pca = PCA(n_components=n_components_search)
        sub_input_search_pca = pd.DataFrame(search_pca.fit_transform(sub_input_search))
        input_test_search_pca = pd.DataFrame(search_pca.transform(input_test_search))
        while np.sum(search_pca.explained_variance_ratio_)<0.80:
            n_components_search=n_components_search+1
            search_pca = PCA(n_components=n_components_search)
            sub_input_search_pca = pd.DataFrame(search_pca.fit_transform(sub_input_search))
            input_test_search_pca = pd.DataFrame(search_pca.transform(input_test_search))    
        sub_inputs_pca=pd.concat([sub_input_volume2,sub_input_search_pca],axis=1)
        input_test_pca=pd.concat([input_test_volume2,input_test_search_pca],axis=1)
        output_scaler,input_fit_n,input_test_n,y=guiyihua(sub_inputs_pca,input_test_pca,y)
        model = pm.auto_arima(y=y,X=input_fit_n,seasonal=True, information_criterion='aic')
        p, d, q = model.order  
        arima_params.append((p, d, q))  
        
        # make your forecasts
        ini_fore = model.predict(1,input_test_n) 
        fin_fore=output_scaler.inverse_transform(ini_fore.reshape(-1,1))
        if fin_fore<0:
            fin_fore=0
        
        arimax_models.append(model)
        arimax_fore.append(float(fin_fore))
        sub_components=[n_components_search]
        n_component_list.append(sub_components)
    n_component_list=np.array(n_component_list)
    arimax_fore=np.array(arimax_fore)
    arimax_end_t=datetime.datetime.now()    
    arimax_run_time=((arimax_end_t-arimax_start_t).seconds)    
    matrix1[step-1]=arimax_fore
    rmse = calculate_rmse(actual, pd.DataFrame(arimax_fore))
    mae = calculate_mae(actual, pd.DataFrame(arimax_fore))  
    ia = calculate_ia(actual, pd.DataFrame(arimax_fore))
    mape = calculate_mape(actual, pd.DataFrame(arimax_fore))  
        
    metrics_results = metrics_results.append({
        'Step': step,
        'Forecast Model': f'Model {fore_mode}',
        'RMSE':rmse,
        'MAE':mae,       
        'IA':ia,
        'MAPE': mape
    }, ignore_index=True)
        
    # 2.volume+review
    fore_mode=2
    input_volume2=pd.concat([volume2],axis=1)   
    input_review=pd.concat([review2],axis=1)
       
    input_volume2=shenzhan(input_volume2,ahead)
    input_review=shenzhan(input_review,ahead)
    arimax_start_t=datetime.datetime.now()
    arimax_models=[]
    arimax_fore=[]
    n_component_list=[]
    for i in range(62):
        y=volume2[ahead+step-1:(len(volume2)-62+i)]
        y=pd.DataFrame(y.values)
        actual=volume2[(len(volume2)-62):len(volume2)]
        actual=pd.DataFrame(actual.values)
        sub_input_volume2=input_volume2[0:(len(volume2)-62+i-step-ahead+1)]
        input_test_volume2=input_volume2[(len(volume2)-62+i-step-ahead+1):(len(volume2)-62+i-step-ahead+2)]
        sub_input_volume2=pd.DataFrame(sub_input_volume2)
        input_test_volume2=pd.DataFrame(input_test_volume2)
        sub_input_review=input_review[0:(len(volume2)-62+i-step-ahead+1)]
        input_test_review=input_review[(len(volume2)-62+i-step-ahead+1):(len(volume2)-62+i-step-ahead+2)]
        n_components_volume2=2
        volume2_pca = PCA(n_components=n_components_volume2)
        sub_input_volume2_pca = pd.DataFrame(volume2_pca.fit_transform(sub_input_volume2))
        input_test_volume2_pca = pd.DataFrame(volume2_pca.transform(input_test_volume2))
        while np.sum(volume2_pca.explained_variance_ratio_)<0.80:
            n_components_volume2=n_components_volume2+1
            volume2_pca = PCA(n_components=n_components_volume2)
            sub_input_volume2_pca = pd.DataFrame(volume2_pca.fit_transform(sub_input_volume2))
            input_test_volume2_pca = pd.DataFrame(volume2_pca.transform(input_test_volume2))

        n_components_review=2
        review_pca = PCA(n_components=n_components_review)
        sub_input_review_pca = pd.DataFrame(review_pca.fit_transform(sub_input_review))
        input_test_review_pca = pd.DataFrame(review_pca.transform(input_test_review))
        while np.sum(review_pca.explained_variance_ratio_)<0.80:
            n_components_review=n_components_review+1
            review_pca = PCA(n_components=n_components_review)
            sub_input_review_pca = pd.DataFrame(review_pca.fit_transform(sub_input_review))
            input_test_review_pca = pd.DataFrame(review_pca.transform(input_test_review))    
        
        sub_inputs_pca=pd.concat([sub_input_volume2,sub_input_review_pca],axis=1)
        input_test_pca=pd.concat([input_test_volume2,input_test_review_pca],axis=1)
        output_scaler,input_fit_n,input_test_n,y=guiyihua(sub_inputs_pca,input_test_pca,y)
        model = pm.auto_arima(y=y,X=input_fit_n,seasonal=True, information_criterion='aic')
        p, d, q = model.order  
        arima_params.append((p, d, q))  
        
        # make your forecasts
        ini_fore = model.predict(1,input_test_n) 
        fin_fore=output_scaler.inverse_transform(ini_fore.reshape(-1,1))
        if fin_fore<0:
            fin_fore=0
        
        arimax_models.append(model)
        arimax_fore.append(float(fin_fore))
        sub_components=[n_components_search,n_components_review]
        n_component_list.append(sub_components)
    n_component_list=np.array(n_component_list)
    arimax_fore=np.array(arimax_fore)
    arimax_end_t=datetime.datetime.now()    
    arimax_run_time=((arimax_end_t-arimax_start_t).seconds)    
    matrix2[step-1]=arimax_fore
    rmse = calculate_rmse(actual, pd.DataFrame(arimax_fore))
    mae = calculate_mae(actual, pd.DataFrame(arimax_fore))   
    ia = calculate_ia(actual, pd.DataFrame(arimax_fore))
    mape = calculate_mape(actual, pd.DataFrame(arimax_fore))  
        
    metrics_results = metrics_results.append({
        'Step': step,
        'Forecast Model': f'Model {fore_mode}',
        'RMSE':rmse,
        'MAE':mae,       
        'IA':ia,
        'MAPE': mape
    }, ignore_index=True)
    
    # 3.volume+weibo
    fore_mode=3
    input_volume2=pd.concat([volume2],axis=1)   
    input_weibo=pd.concat([weibo2],axis=1)
        
    input_volume2=shenzhan(input_volume2,ahead)
    input_weibo=shenzhan(input_weibo,ahead)
    arimax_start_t=datetime.datetime.now()
    arimax_models=[]
    arimax_fore=[]
    n_component_list=[]
    for i in range(62):
        y=volume2[ahead+step-1:(len(volume2)-62+i)]
        y=pd.DataFrame(y.values)
        actual=volume2[(len(volume2)-62):len(volume2)]
        actual=pd.DataFrame(actual.values)
        sub_input_volume2=input_volume2[0:(len(volume2)-62+i-step-ahead+1)]
        input_test_volume2=input_volume2[(len(volume2)-62+i-step-ahead+1):(len(volume2)-62+i-step-ahead+2)]
        sub_input_volume2=pd.DataFrame(sub_input_volume2)
        input_test_volume2=pd.DataFrame(input_test_volume2)
        sub_input_weibo=input_weibo[0:(len(volume2)-62+i-step-ahead+1)]
        input_test_weibo=input_weibo[(len(volume2)-62+i-step-ahead+1):(len(volume2)-62+i-step-ahead+2)]
        n_components_volume2=2
        volume2_pca = PCA(n_components=n_components_volume2)
        sub_input_volume2_pca = pd.DataFrame(volume2_pca.fit_transform(sub_input_volume2))
        input_test_volume2_pca = pd.DataFrame(volume2_pca.transform(input_test_volume2))
        while np.sum(volume2_pca.explained_variance_ratio_)<0.80:
            n_components_volume2=n_components_volume2+1
            volume2_pca = PCA(n_components=n_components_volume2)
            sub_input_volume2_pca = pd.DataFrame(volume2_pca.fit_transform(sub_input_volume2))
            input_test_volume2_pca = pd.DataFrame(volume2_pca.transform(input_test_volume2))
        n_components_weibo=2
        weibo_pca = PCA(n_components=n_components_weibo)
        sub_input_weibo_pca = pd.DataFrame(weibo_pca.fit_transform(sub_input_weibo))
        input_test_weibo_pca = pd.DataFrame(weibo_pca.transform(input_test_weibo))
        while np.sum(weibo_pca.explained_variance_ratio_)<0.80:
            n_components_weibo=n_components_weibo+1
            weibo_pca = PCA(n_components=n_components_weibo)
            sub_input_weibo_pca = pd.DataFrame(weibo_pca.fit_transform(sub_input_weibo))
            input_test_weibo_pca = pd.DataFrame(weibo_pca.transform(input_test_weibo))    
        sub_inputs_pca=pd.concat([sub_input_volume2,sub_input_weibo_pca],axis=1)
        input_test_pca=pd.concat([input_test_volume2,input_test_weibo_pca],axis=1)
        output_scaler,input_fit_n,input_test_n,y=guiyihua(sub_inputs_pca,input_test_pca,y)
        model = pm.auto_arima(y=y,X=input_fit_n,seasonal=True, information_criterion='aic')
        p, d, q = model.order  
        arima_params.append((p, d, q))  
        
        # make your forecasts
        ini_fore = model.predict(1,input_test_n) 
        fin_fore=output_scaler.inverse_transform(ini_fore.reshape(-1,1))
        if fin_fore<0:
            fin_fore=0
        
        arimax_models.append(model)
        arimax_fore.append(float(fin_fore))
        sub_components=[n_components_weibo]
        n_component_list.append(sub_components)
    n_component_list=np.array(n_component_list)
    arimax_fore=np.array(arimax_fore)
    arimax_end_t=datetime.datetime.now()    
    arimax_run_time=((arimax_end_t-arimax_start_t).seconds)    
    matrix3[step-1]=arimax_fore
    rmse = calculate_rmse(actual, pd.DataFrame(arimax_fore))
    mae = calculate_mae(actual, pd.DataFrame(arimax_fore))    
    ia = calculate_ia(actual, pd.DataFrame(arimax_fore))
    mape = calculate_mape(actual, pd.DataFrame(arimax_fore))  
        
    metrics_results = metrics_results.append({
        'Step': step,
        'Forecast Model': f'Model {fore_mode}',
        'RMSE':rmse,
        'MAE':mae,       
        'IA':ia,
        'MAPE': mape
    }, ignore_index=True)
    
    # 4.volume+search+weibo
    fore_mode=4
    input_volume2=pd.concat([volume2],axis=1) 
    input_search=pd.concat([search2],axis=1)   
    input_weibo=pd.concat([weibo2],axis=1)
        
    input_volume2=shenzhan(input_volume2,ahead)
    input_search=shenzhan(input_search,ahead)
    input_weibo=shenzhan(input_weibo,ahead)
    arimax_start_t=datetime.datetime.now()
    arimax_models=[]
    arimax_fore=[]
    n_component_list=[]
    for i in range(62):
        y=volume2[ahead+step-1:(len(volume2)-62+i)]
        y=pd.DataFrame(y.values)
        actual=volume2[(len(volume2)-62):len(volume2)]
        actual=pd.DataFrame(actual.values)
        sub_input_volume2=input_volume2[0:(len(volume2)-62+i-step-ahead+1)]
        input_test_volume2=input_volume2[(len(volume2)-62+i-step-ahead+1):(len(volume2)-62+i-step-ahead+2)]
        sub_input_volume2=pd.DataFrame(sub_input_volume2)
        input_test_volume2=pd.DataFrame(input_test_volume2)
        sub_input_search=input_search[0:(len(volume2)-62+i-step-ahead+1)]
        input_test_search=input_search[(len(volume2)-62+i-step-ahead+1):(len(volume2)-62+i-step-ahead+2)]
        sub_input_weibo=input_weibo[0:(len(volume2)-62+i-step-ahead+1)]
        input_test_weibo=input_weibo[(len(volume2)-62+i-step-ahead+1):(len(volume2)-62+i-step-ahead+2)]
        n_components_volume2=2
        volume2_pca = PCA(n_components=n_components_volume2)
        sub_input_volume2_pca = pd.DataFrame(volume2_pca.fit_transform(sub_input_volume2))
        input_test_volume2_pca = pd.DataFrame(volume2_pca.transform(input_test_volume2))
        while np.sum(volume2_pca.explained_variance_ratio_)<0.80:
            n_components_volume2=n_components_volume2+1
            volume2_pca = PCA(n_components=n_components_volume2)
            sub_input_volume2_pca = pd.DataFrame(volume2_pca.fit_transform(sub_input_volume2))
            input_test_volume2_pca = pd.DataFrame(volume2_pca.transform(input_test_volume2))

        n_components_search=2
        search_pca = PCA(n_components=n_components_search)
        sub_input_search_pca = pd.DataFrame(search_pca.fit_transform(sub_input_search))
        input_test_search_pca = pd.DataFrame(search_pca.transform(input_test_search))
        while np.sum(search_pca.explained_variance_ratio_)<0.80:
            n_components_search=n_components_search+1
            search_pca = PCA(n_components=n_components_search)
            sub_input_search_pca = pd.DataFrame(search_pca.fit_transform(sub_input_search))
            input_test_search_pca = pd.DataFrame(search_pca.transform(input_test_search))
        n_components_weibo=2
        weibo_pca = PCA(n_components=n_components_weibo)
        sub_input_weibo_pca = pd.DataFrame(weibo_pca.fit_transform(sub_input_weibo))
        input_test_weibo_pca = pd.DataFrame(weibo_pca.transform(input_test_weibo))
        while np.sum(weibo_pca.explained_variance_ratio_)<0.80:
            n_components_weibo=n_components_weibo+1
            weibo_pca = PCA(n_components=n_components_weibo)
            sub_input_weibo_pca = pd.DataFrame(weibo_pca.fit_transform(sub_input_weibo))
            input_test_weibo_pca = pd.DataFrame(weibo_pca.transform(input_test_weibo))    
        sub_inputs_pca=pd.concat([sub_input_volume2,sub_input_search_pca,sub_input_weibo_pca],axis=1)
        input_test_pca=pd.concat([input_test_volume2,input_test_search_pca,input_test_weibo_pca],axis=1)
        output_scaler,input_fit_n,input_test_n,y=guiyihua(sub_inputs_pca,input_test_pca,y)
        model = pm.auto_arima(y=y,X=input_fit_n,seasonal=True, information_criterion='aic')
        p, d, q = model.order  
        arima_params.append((p, d, q))  
        
        # make your forecasts
        ini_fore = model.predict(1,input_test_n) 
        fin_fore=output_scaler.inverse_transform(ini_fore.reshape(-1,1))
        if fin_fore<0:
            fin_fore=0
        
        arimax_models.append(model)
        arimax_fore.append(float(fin_fore))
        sub_components=[n_components_search,n_components_weibo]
        n_component_list.append(sub_components)
    n_component_list=np.array(n_component_list)
    arimax_fore=np.array(arimax_fore)
    arimax_end_t=datetime.datetime.now()    
    arimax_run_time=((arimax_end_t-arimax_start_t).seconds)    
    matrix4[step-1]=arimax_fore
    rmse = calculate_rmse(actual, pd.DataFrame(arimax_fore))
    mae = calculate_mae(actual, pd.DataFrame(arimax_fore))
    
    ia = calculate_ia(actual, pd.DataFrame(arimax_fore))
    mape = calculate_mape(actual, pd.DataFrame(arimax_fore))  
        
    metrics_results = metrics_results.append({
        'Step': step,
        'Forecast Model': f'Model {fore_mode}',
        'RMSE':rmse,
        'MAE':mae,  
        'IA':ia,
        'MAPE': mape
    }, ignore_index=True)
    
    # 5.volume+review+weibo
    fore_mode=5
    input_volume2=pd.concat([volume2],axis=1)
    input_review=pd.concat([review2],axis=1)
    input_weibo=pd.concat([weibo2],axis=1)
        
    input_volume2=shenzhan(input_volume2,ahead)
    input_review=shenzhan(input_review,ahead)
    input_weibo=shenzhan(input_weibo,ahead)
    arimax_start_t=datetime.datetime.now()
    arimax_models=[]
    arimax_fore=[]
    n_component_list=[]
    for i in range(62):
        y=volume2[ahead+step-1:(len(volume2)-62+i)]
        y=pd.DataFrame(y.values)
        actual=volume2[(len(volume2)-62):len(volume2)]
        actual=pd.DataFrame(actual.values)
        sub_input_volume2=input_volume2[0:(len(volume2)-62+i-step-ahead+1)]
        input_test_volume2=input_volume2[(len(volume2)-62+i-step-ahead+1):(len(volume2)-62+i-step-ahead+2)]
        sub_input_volume2=pd.DataFrame(sub_input_volume2)
        input_test_volume2=pd.DataFrame(input_test_volume2)
        sub_input_review=input_review[0:(len(volume2)-62+i-step-ahead+1)]
        input_test_review=input_review[(len(volume2)-62+i-step-ahead+1):(len(volume2)-62+i-step-ahead+2)]
        sub_input_weibo=input_weibo[0:(len(volume2)-62+i-step-ahead+1)]
        input_test_weibo=input_weibo[(len(volume2)-62+i-step-ahead+1):(len(volume2)-62+i-step-ahead+2)]
        n_components_volume2=2
        volume2_pca = PCA(n_components=n_components_volume2)
        sub_input_volume2_pca = pd.DataFrame(volume2_pca.fit_transform(sub_input_volume2))
        input_test_volume2_pca = pd.DataFrame(volume2_pca.transform(input_test_volume2))
        while np.sum(volume2_pca.explained_variance_ratio_)<0.80:
            n_components_volume2=n_components_volume2+1
            volume2_pca = PCA(n_components=n_components_volume2)
            sub_input_volume2_pca = pd.DataFrame(volume2_pca.fit_transform(sub_input_volume2))
            input_test_volume2_pca = pd.DataFrame(volume2_pca.transform(input_test_volume2))

        n_components_review=2
        review_pca = PCA(n_components=n_components_review)
        sub_input_review_pca = pd.DataFrame(review_pca.fit_transform(sub_input_review))
        input_test_review_pca = pd.DataFrame(review_pca.transform(input_test_review))
        while np.sum(review_pca.explained_variance_ratio_)<0.80:
            n_components_review=n_components_review+1
            review_pca = PCA(n_components=n_components_review)
            sub_input_review_pca = pd.DataFrame(review_pca.fit_transform(sub_input_review))
            input_test_review_pca = pd.DataFrame(review_pca.transform(input_test_review))
        n_components_weibo=2
        weibo_pca = PCA(n_components=n_components_weibo)
        sub_input_weibo_pca = pd.DataFrame(weibo_pca.fit_transform(sub_input_weibo))
        input_test_weibo_pca = pd.DataFrame(weibo_pca.transform(input_test_weibo))
        while np.sum(weibo_pca.explained_variance_ratio_)<0.80:
            n_components_weibo=n_components_weibo+1
            weibo_pca = PCA(n_components=n_components_weibo)
            sub_input_weibo_pca = pd.DataFrame(weibo_pca.fit_transform(sub_input_weibo))
            input_test_weibo_pca = pd.DataFrame(weibo_pca.transform(input_test_weibo))
        sub_inputs_pca=pd.concat([sub_input_volume2,sub_input_review_pca,sub_input_weibo_pca],axis=1)
        input_test_pca=pd.concat([input_test_volume2,input_test_review_pca,input_test_weibo_pca],axis=1)
        output_scaler,input_fit_n,input_test_n,y=guiyihua(sub_inputs_pca,input_test_pca,y)
        model = pm.auto_arima(y=y,X=input_fit_n,seasonal=True, information_criterion='aic')
        p, d, q = model.order  
        arima_params.append((p, d, q))  
        
        # make your forecasts
        ini_fore = model.predict(1,input_test_n) 
        fin_fore=output_scaler.inverse_transform(ini_fore.reshape(-1,1))
        if fin_fore<0:
            fin_fore=0
        
        arimax_models.append(model)
        arimax_fore.append(float(fin_fore))
        sub_components=[n_components_review,n_components_weibo]
        n_component_list.append(sub_components)
    n_component_list=np.array(n_component_list)
    arimax_fore=np.array(arimax_fore)
    arimax_end_t=datetime.datetime.now()    
    arimax_run_time=((arimax_end_t-arimax_start_t).seconds)
    matrix5[step-1]=arimax_fore
    rmse = calculate_rmse(actual, pd.DataFrame(arimax_fore))
    mae = calculate_mae(actual, pd.DataFrame(arimax_fore))   
    ia = calculate_ia(actual, pd.DataFrame(arimax_fore))
    mape = calculate_mape(actual, pd.DataFrame(arimax_fore))  
        
    metrics_results = metrics_results.append({
        'Step': step,
        'Forecast Model': f'Model {fore_mode}',
        'RMSE':rmse,
        'MAE':mae,       
        'IA':ia,
        'MAPE': mape
    }, ignore_index=True)
    
    # 6.volume+review+search
    fore_mode=6
    input_volume2=pd.concat([volume2],axis=1)
    input_review=pd.concat([review2],axis=1)  
    input_search=pd.concat([search2],axis=1)
       
    input_volume2=shenzhan(input_volume2,ahead)
    input_review=shenzhan(input_review,ahead)
    input_search=shenzhan(input_search,ahead)
    arimax_start_t=datetime.datetime.now()
    arimax_models=[]
    arimax_fore=[]
    n_component_list=[]
    for i in range(62):
        y=volume2[ahead+step-1:(len(volume2)-62+i)]
        y=pd.DataFrame(y.values)
        actual=volume2[(len(volume2)-62):len(volume2)]
        actual=pd.DataFrame(actual.values)
        sub_input_volume2=input_volume2[0:(len(volume2)-62+i-step-ahead+1)]
        input_test_volume2=input_volume2[(len(volume2)-62+i-step-ahead+1):(len(volume2)-62+i-step-ahead+2)]
        sub_input_volume2=pd.DataFrame(sub_input_volume2)
        input_test_volume2=pd.DataFrame(input_test_volume2)
        sub_input_review=input_review[0:(len(volume2)-62+i-step-ahead+1)]
        input_test_review=input_review[(len(volume2)-62+i-step-ahead+1):(len(volume2)-62+i-step-ahead+2)]
        sub_input_search=input_search[0:(len(volume2)-62+i-step-ahead+1)]
        input_test_search=input_search[(len(volume2)-62+i-step-ahead+1):(len(volume2)-62+i-step-ahead+2)]
        
        n_components_volume2=2
        volume2_pca = PCA(n_components=n_components_volume2)
        sub_input_volume2_pca = pd.DataFrame(volume2_pca.fit_transform(sub_input_volume2))
        input_test_volume2_pca = pd.DataFrame(volume2_pca.transform(input_test_volume2))
        while np.sum(volume2_pca.explained_variance_ratio_)<0.80:
            n_components_volume2=n_components_volume2+1
            volume2_pca = PCA(n_components=n_components_volume2)
            sub_input_volume2_pca = pd.DataFrame(volume2_pca.fit_transform(sub_input_volume2))
            input_test_volume2_pca = pd.DataFrame(volume2_pca.transform(input_test_volume2))

        n_components_review=2
        review_pca = PCA(n_components=n_components_review)
        sub_input_review_pca = pd.DataFrame(review_pca.fit_transform(sub_input_review))
        input_test_review_pca = pd.DataFrame(review_pca.transform(input_test_review))
        while np.sum(review_pca.explained_variance_ratio_)<0.80:
            n_components_review=n_components_review+1
            review_pca = PCA(n_components=n_components_review)
            sub_input_review_pca = pd.DataFrame(review_pca.fit_transform(sub_input_review))
            input_test_review_pca = pd.DataFrame(review_pca.transform(input_test_review))
        
        n_components_search=2
        search_pca = PCA(n_components=n_components_search)
        sub_input_search_pca = pd.DataFrame(search_pca.fit_transform(sub_input_search))
        input_test_search_pca = pd.DataFrame(search_pca.transform(input_test_search))
        while np.sum(search_pca.explained_variance_ratio_)<0.80:
            n_components_search=n_components_search+1
            search_pca = PCA(n_components=n_components_search)
            sub_input_search_pca = pd.DataFrame(search_pca.fit_transform(sub_input_search))
            input_test_search_pca = pd.DataFrame(search_pca.transform(input_test_search))
        sub_inputs_pca=pd.concat([sub_input_volume2,sub_input_review_pca,sub_input_search_pca],axis=1)
        input_test_pca=pd.concat([input_test_volume2,input_test_review_pca,input_test_search_pca],axis=1)
        output_scaler,input_fit_n,input_test_n,y=guiyihua(sub_inputs_pca,input_test_pca,y)
        model = pm.auto_arima(y=y,X=input_fit_n,seasonal=True, information_criterion='aic')
        p, d, q = model.order  
        arima_params.append((p, d, q))  
        
        # make your forecasts
        ini_fore = model.predict(1,input_test_n) 
        fin_fore=output_scaler.inverse_transform(ini_fore.reshape(-1,1))
        if fin_fore<0:
            fin_fore=0
        
        arimax_models.append(model)
        arimax_fore.append(float(fin_fore))
        sub_components=[n_components_review,n_components_search]
        n_component_list.append(sub_components)
    n_component_list=np.array(n_component_list)
    arimax_fore=np.array(arimax_fore)
    arimax_end_t=datetime.datetime.now()    
    arimax_run_time=((arimax_end_t-arimax_start_t).seconds)    
    matrix6[step-1]=arimax_fore
    rmse = calculate_rmse(actual, pd.DataFrame(arimax_fore))
    mae = calculate_mae(actual, pd.DataFrame(arimax_fore)) 
    ia = calculate_ia(actual, pd.DataFrame(arimax_fore))
    mape = calculate_mape(actual, pd.DataFrame(arimax_fore))  
        
    metrics_results = metrics_results.append({
        'Step': step,
        'Forecast Model': f'Model {fore_mode}',
        'RMSE':rmse,
        'MAE':mae,      
        'IA':ia,
        'MAPE': mape
    }, ignore_index=True)
    
    # 7.volume+search+review+weibo
    fore_mode=7
    input_volume2=pd.concat([volume2],axis=1)    
    input_search=pd.concat([search2],axis=1)    
    input_review=pd.concat([review2],axis=1)
    input_weibo=pd.concat([weibo2],axis=1)
        
    input_volume2=shenzhan(input_volume2,ahead)
    input_search=shenzhan(input_search,ahead)
    input_review=shenzhan(input_review,ahead)
    input_weibo=shenzhan(input_weibo,ahead)
    arimax_start_t=datetime.datetime.now()
    arimax_models=[]
    arimax_fore=[]
    n_component_list=[]
    for i in range(62):
        y=volume2[ahead+step-1:(len(volume2)-62+i)]
        y=pd.DataFrame(y.values)
        actual=volume2[(len(volume2)-62):len(volume2)]
        actual=pd.DataFrame(actual.values)
        sub_input_volume2=input_volume2[0:(len(volume2)-62+i-step-ahead+1)]
        input_test_volume2=input_volume2[(len(volume2)-62+i-step-ahead+1):(len(volume2)-62+i-step-ahead+2)]
        sub_input_volume2=pd.DataFrame(sub_input_volume2)
        input_test_volume2=pd.DataFrame(input_test_volume2)
        
        sub_input_search=input_search[0:(len(volume2)-62+i-step-ahead+1)]
        input_test_search=input_search[(len(volume2)-62+i-step-ahead+1):(len(volume2)-62+i-step-ahead+2)]
        sub_input_review=input_review[0:(len(volume2)-62+i-step-ahead+1)]
        input_test_review=input_review[(len(volume2)-62+i-step-ahead+1):(len(volume2)-62+i-step-ahead+2)]
        sub_input_weibo=input_weibo[0:(len(volume2)-62+i-step-ahead+1)]
        input_test_weibo=input_weibo[(len(volume2)-62+i-step-ahead+1):(len(volume2)-62+i-step-ahead+2)]
        
        n_components_search=2
        search_pca = PCA(n_components=n_components_search)
        sub_input_search_pca = pd.DataFrame(search_pca.fit_transform(sub_input_search))
        input_test_search_pca = pd.DataFrame(search_pca.transform(input_test_search))
        while np.sum(search_pca.explained_variance_ratio_)<0.80:
            n_components_search=n_components_search+1
            search_pca = PCA(n_components=n_components_search)
            sub_input_search_pca = pd.DataFrame(search_pca.fit_transform(sub_input_search))
            input_test_search_pca = pd.DataFrame(search_pca.transform(input_test_search))
        n_components_review=2
        review_pca = PCA(n_components=n_components_review)
        sub_input_review_pca = pd.DataFrame(review_pca.fit_transform(sub_input_review))
        input_test_review_pca = pd.DataFrame(review_pca.transform(input_test_review))
        while np.sum(review_pca.explained_variance_ratio_)<0.80:
            n_components_review=n_components_review+1
            review_pca = PCA(n_components=n_components_review)
            sub_input_review_pca = pd.DataFrame(review_pca.fit_transform(sub_input_review))
            input_test_review_pca = pd.DataFrame(review_pca.transform(input_test_review))
        n_components_weibo=2
        weibo_pca = PCA(n_components=n_components_weibo)
        sub_input_weibo_pca = pd.DataFrame(weibo_pca.fit_transform(sub_input_weibo))
        input_test_weibo_pca = pd.DataFrame(weibo_pca.transform(input_test_weibo))
        while np.sum(weibo_pca.explained_variance_ratio_)<0.80:
            n_components_weibo=n_components_weibo+1
            weibo_pca = PCA(n_components=n_components_weibo)
            sub_input_weibo_pca = pd.DataFrame(weibo_pca.fit_transform(sub_input_weibo))
            input_test_weibo_pca = pd.DataFrame(weibo_pca.transform(input_test_weibo))
        sub_inputs_pca=pd.concat([sub_input_search_pca,sub_input_review_pca,sub_input_weibo_pca],axis=1)
        input_test_pca=pd.concat([input_test_search_pca,input_test_review_pca,input_test_weibo_pca],axis=1)
        output_scaler,input_fit_n,input_test_n,y=guiyihua(sub_inputs_pca,input_test_pca,y)
        model = pm.auto_arima(y=y,X=input_fit_n,seasonal=True, information_criterion='aic')
        p, d, q = model.order  
        arima_params.append((p, d, q))  
        
        # make your forecasts
        ini_fore = model.predict(1,input_test_n) 
        fin_fore=output_scaler.inverse_transform(ini_fore.reshape(-1,1))
        if fin_fore<0:
            fin_fore=0
        
        arimax_models.append(model)
        arimax_fore.append(float(fin_fore))
        sub_components=[n_components_search,n_components_review,n_components_weibo]
        n_component_list.append(sub_components)
    n_component_list=np.array(n_component_list)
    arimax_fore=np.array(arimax_fore)
    arimax_end_t=datetime.datetime.now()    
    arimax_run_time=((arimax_end_t-arimax_start_t).seconds)    
    matrix7[step-1]=arimax_fore
    rmse = calculate_rmse(actual, pd.DataFrame(arimax_fore))
    mae = calculate_mae(actual, pd.DataFrame(arimax_fore)) 
    ia = calculate_ia(actual, pd.DataFrame(arimax_fore))
    mape = calculate_mape(actual, pd.DataFrame(arimax_fore))  
        
    metrics_results = metrics_results.append({
        'Step': step,
        'Forecast Model': f'Model {fore_mode}',
        'RMSE':rmse,
        'MAE':mae,      
        'IA':ia,
        'MAPE': mape
    }, ignore_index=True)
    
df_params1 = pd.DataFrame(arima_params, columns=['p', 'd', 'q'])

matrix31=np.array([matrix1[0],matrix2[0],matrix3[0],matrix4[0],matrix5[0],matrix6[0],matrix7[0]])
matrix32=np.array([matrix1[1],matrix2[1],matrix3[1],matrix4[1],matrix5[1],matrix6[1],matrix7[1]])
matrix33=np.array([matrix1[2],matrix2[2],matrix3[2],matrix4[2],matrix5[2],matrix6[2],matrix7[2]])

matrix31=matrix31.T
matrix32=matrix32.T
matrix33=matrix33.T