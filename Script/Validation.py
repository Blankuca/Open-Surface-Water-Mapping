# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 17:14:54 2020

@author: remot
"""
from Script.Processing import make_dataset, add_indices, NDWI, AWEI
import Script.Image as Image
import pandas as pd
from scipy import ndimage
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score

x_cols = ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12','B13']

#%%

# Function to evaluate multi-class predictions
def evaluate(y_true, y_pred, print_cm=False, print_err=False):
    
    # calculate and display confusion matrix
    labels = np.unique(y_true)
    names = ['Deep water', 'Shallow water', 'Dry/Urban', 'Soil', 'Vegetated land']
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if print_cm:
        ax= plt.subplot()
        sn.heatmap(cm, annot=True, ax = ax, xticklabels=names, yticklabels=names, cmap = 'Reds', fmt = 'd')
        ax.set_xlabel('Predicted');
        ax.set_ylabel('True'); 

    # calculate precision, recall, and F1 score
    accuracy = float(np.trace(cm)) / np.sum(cm)
    precision = precision_score(y_true, y_pred, average=None, labels=labels)[1]
    recall = recall_score(y_true, y_pred, average=None, labels=labels)[1]
    f1 = 2 * precision * recall / (precision + recall)
    if print_err:
        print("")
        print("accuracy:", accuracy)
        print("precision:", precision)
        print("recall:", recall)
        print("f1 score:", f1)
        
    return [accuracy,precision,recall,f1]

# Function to evaluate water/ non-water predictions
def binary(y_true, y_pred):
    y_pred = [1 if x in [1,2] else 0 for x in y_pred]
    y_true = [1 if x in [1,2] else 0 for x in y_true]
    labels = np.unique(y_true)
    names = ['Non-water', 'Water']
    ax= plt.subplot()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sn.heatmap(cm, annot=True, ax=ax,xticklabels=names, yticklabels=names, cmap = 'Reds', fmt = 'd')
    ax.set_xlabel('Predicted');
    ax.set_ylabel('True'); 
    return evaluate(y_true, y_pred, print_err= True)

#%%  Cross-validations
    
# K-fold cross-validation for finding optimal lambda, which minimizes error
def cross_val(training,K):

        lambda_interval = np.power(10.,range(-7,7))
        
        CV = model_selection.KFold(n_splits = K, shuffle = False)
        
        gen_acc = np.zeros(len(lambda_interval))
        
        n = len(lambda_interval)
        
        for s in range(0, n):
            k = 0
            
            val_acc = np.zeros(K)
            
            X = training[x_cols].values
            y = training['Content'].values
            
            for train_index, val_index in CV.split(X, y):
    
                # extract training and test set for current CV fold
                X_train, y_train = X[train_index,:], y[train_index]
                X_val, y_val = X[val_index,:], y[val_index]
            
                model = LogisticRegression(penalty = 'l2', C = 1/lambda_interval[s], solver = 'lbfgs', multi_class='multinomial') 
                model = model.fit(X_train, y_train)
    
                y_val_estimated = model.predict(X_val).T
                
                val_acc[k] = np.mean( y_val != y_val_estimated)*100
                
                k += 1
            
            gen_acc[s] = np.sum(val_acc) / len(val_acc)
            print('Iteration number {0}/{1}. Error is {2}'.format(s,n, gen_acc[s]))
            
        max_acc = np.min(gen_acc)       
        opt_lambda_index = np.argmin(gen_acc) 
        opt_lambda = lambda_interval[opt_lambda_index]
       
        print('Accuracy - regularized log-reg - {0}'.format(np.round(max_acc, decimals = 3)))
        print('Optimal lambda: {0}'.format(opt_lambda))
        print()
        
        plt.plot(np.log10(lambda_interval),gen_acc, color = 'skyblue')
        plt.xlabel('Log10(Lambda)')
        plt.ylabel('Error')
        plt.grid()


# Find optimal number of clusters for K-means to minimize error
def optimal_cluster(model, root, rasterpath):
    n_clusters = list(range(5,15))
    error = np.zeros(len(n_clusters))
    
    X = Image.make_dataset(rasterpath)['df']
    y_pred = model.predict(X)
    
    y_true = make_dataset(root, rasterpath).Content
       
    for n in range(len(n_clusters)):       
        y_pred = pd.DataFrame(ndimage.median_filter(y_pred, size=n_clusters[n]))
        y_pred = y_pred.loc[y_true.index]
        error[n] = np.mean(evaluate(y_true, y_pred))
    
    min_error = np.min(error)
    opt_index = np.argmin(error)
    opt_clusters = n_clusters[opt_index]
    
    print('Min error is {0}, using {1} clusters'.format(min_error, opt_clusters))
    
    plt.plot(n_clusters,error)
    plt.xlabel('Number of clusters')
    plt.ylabel('Error')
    
#%%   
    
# Get predictions of a validation dataset for a model and the true value
def y_true_pred(model, root, rasterpath): 
    
    df = make_dataset(root, rasterpath) 

    y_true = df.Content
    y_true = [3 if x==6 else x for x in y_true]
    X = df[x_cols]
    y_pred = model.predict(X)

            
    return y_true, y_pred

# Get predictions of the complete validation dataset for a model and the true value
def all_y_true_pred(model, area, dates):
    y_true = np.array([])
    y_pred = np.array([])
    
    for date in dates:
        
        if area == 'NW':
            root = 'H:/sentinel2/Ghana/Validation/' + date + '/NW/Shapefiles/'
            rasterpath = 'H:/sentinel2/Ghana/Validation/' + date + '/NW/Agona.tif'
            
        elif area == 'SE':
            root = 'H:/sentinel2/Ghana/Validation/' + date + '/SE/Shapefiles/'
            rasterpath = 'H:/sentinel2/Ghana/Validation/' + date + '/SE/Obom.tif'
        
        yt, yp = y_true_pred(model, root, rasterpath)
        y_true = np.append(y_true,yt)
        y_pred = np.append(y_pred,yp)
        
        y_true[0] = 1
        y_pred[0] = 1
                
    return y_true, y_pred

# Get predictions of a validation dataset for a water index and the true value
def y_true_pred_index(index, root, rasterpath, t =0): 
    
    df = make_dataset(root, rasterpath) 

    y_true = df.Content
    y_true = [1 if x<3 else 0 for x in y_true]
    X = df[x_cols]
    X.columns = ['B1','B2','B3','B4','B5','B6','B7','B8', 'B8A', 'B9','B10','B11','B12']
    if index == 'NDWI':
        _,y_pred = NDWI(X, t)
    elif index == 'AWEI':
        _,y_pred = AWEI(X, t)
            
    return y_true, y_pred

# Get predictions of the complete validation dataset for a water index and the true value
def all_y_true_pred_index(index, area, dates):
    y_true = np.array([])
    y_pred = np.array([])
    
    for date in dates:
        
        if area == 'NW':
            root = 'H:/sentinel2/Ghana/Validation/' + date + '/NW/Shapefiles/'
            rasterpath = 'H:/sentinel2/Ghana/Validation/' + date + '/NW/Agona.tif'
            
        elif area == 'SE':
            root = 'H:/sentinel2/Ghana/Validation/' + date + '/SE/Shapefiles/'
            rasterpath = 'H:/sentinel2/Ghana/Validation/' + date + '/SE/Obom.tif'
        
        yt, yp = y_true_pred_index(index, root, rasterpath)
        y_true = np.append(y_true,yt)
        y_pred = np.append(y_pred,yp)
                
    return y_true, y_pred

    


