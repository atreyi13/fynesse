# This file contains code for suporting addressing questions in the data
from .config import *

from . import access
from . import assess

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""

from datetime import datetime
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import warnings


def hello_world():
    print("This is my first pip package!")

def predict_data_full(latitude, longitude, date, property_type, conn): # This function gets data for prediction and calculates all four vectors
    latstart = latitude - 0.1
    latend = latitude + 0.1
    longstart = longitude - 0.1
    longend = longitude + 0.1
    year = date[-4:]
    datestart =  year+'-01-01'
    dateend = year+'-12-31'
    datedate = datetime.strptime(date, "%d%m%Y").date()
    access.create_price_coord_data(conn)
    df = access.access_for_prediction(latstart, latend, longstart, longend, datestart,dateend, property_type,conn)
    df = df.sort_values(by=['price'])
    pois = assess.get_pois(df)
    test = df[(df['lattitude'] == latitude) & (df['longitude']	== longitude ) & (df['date_of_transfer'] == datedate)]
    df = df[(df['lattitude'] != latitude) | (df['longitude']	!= longitude ) | (df['date_of_transfer'] != datedate)]
    if len(test) == 0:
        data = [[0, datetime.strptime(date, "%d%m%Y").date(), 'postcode', property_type, 'new_build_flag', 'tenure_type', 'locality', 'town/city', 'district', 'county', 'country', latitude, longitude]]
        test = pd.DataFrame(data, columns=['price', 'date_of_transfer', 'postcode', 'property_type'	,'new_build_flag', 'tenure_type', 'locality', 'town_city', 'district', 'county', 'country', 'lattitude', 'longitude' ])
    df['vector_distance_cat'] = assess.vec_app(df,0.02, pois, assess.get_vector_inv_cat)
    df['vector_distance'] = assess.vec_app(df,0.02, pois, assess.get_vector_distance)
    df['vector_count'] = assess.vec_app(df,0.02, pois, assess.get_vector_count)
    df['vector_count_cat'] = assess.vec_app(df, 0.02,pois, assess.get_vector_count_cat)
    test['vector_distance_cat'] = assess.vec_app(test,0.02, pois, assess.get_vector_inv_cat)
    test['vector_distance'] = assess.vec_app(test,0.02, pois, assess.get_vector_distance)
    test['vector_count'] = assess.vec_app(test,0.02, pois, assess.get_vector_count)
    test['vector_count_cat'] = assess.vec_app(test, 0.02,pois, assess.get_vector_count_cat)
    
    return df, test

def predict_data(latitude, longitude, date, property_type, conn, funcname= assess.get_vector_inv_cat): # This function gets data for prediction and calculates all one vector
    latstart = latitude - 0.1
    latstart = latitude - 0.2
    latend = latitude + 0.2
    longstart = longitude - 0.2
    longend = longitude + 0.2
    year = date[-4:]
    datestart =  year+'-01-01'
    dateend = year+'-12-31'
    datedate = datetime.strptime(date, "%d%m%Y").date()
    access.create_price_coord_data(conn)
    df = access.access_for_prediction(latstart, latend, longstart, longend, datestart,dateend, property_type,conn)
    df = df.sort_values(by=['price'])
    pois = assess.get_pois(df)
    test = df[(df['lattitude'] == latitude) & (df['longitude']	== longitude ) & (df['date_of_transfer'] == datedate)]
    df = df[(df['lattitude'] != latitude) | (df['longitude']	!= longitude ) | (df['date_of_transfer'] != datedate)]
    if len(test) == 0:
        data = [[0, datetime.strptime(date, "%d%m%Y").date(), 'postcode', property_type, 'new_build_flag', 'tenure_type', 'locality', 'town/city', 'district', 'county', 'country', latitude, longitude]]
        test = pd.DataFrame(data, columns=['price', 'date_of_transfer', 'postcode', 'property_type'	,'new_build_flag', 'tenure_type', 'locality', 'town_city', 'district', 'county', 'country', 'lattitude', 'longitude' ])
    df['vector'] = assess.vec_app(df,0.02, pois, funcname)
    test['vector'] = assess.vec_app(test,0.02, pois, funcname)
    
    return df, test

  
def eigen_view(df2, column='vector'): # This function plots the eigenvalues
  vector = np.array([np.array(xi) for xi in list(df2[column])])
  data = pd.DataFrame(vector)
  x_meaned = data - np.mean(data , axis = 0)
  cov_mat = np.cov(x_meaned , rowvar = False)
  eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
  sorted_index = np.argsort(eigen_values)[::-1]
  sorted_eigenvalue = eigen_values[sorted_index] 
  sorted_eigenvectors = eigen_vectors[:,sorted_index]
  plt.bar(range(len(sorted_eigenvalue)), np.log(sorted_eigenvalue), color ='green',
        width = 0.4)
  
def prin_comp(components, df2, test, column='vector'): # this function does pca
    vector = np.array([np.array(xi) for xi in list(df2[column])])
    data = pd.DataFrame(vector)
    testvec = np.array([np.array(xi) for xi in list(test[column])])
    testdata = pd.DataFrame(testvec)

    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    testdata = scaler.transform(testdata)

    pca = PCA(n_components=components)
    pca.fit(data)
    data = pca.transform(data)
    testdata = pca.transform(testdata)
    return data, testdata
  
def predict_z_models(df2, test, components,column= 'vector'): #this function does the actual prediction and returns the best model
    data, testdata = prin_comp(components, df2, test, column)
    z = np.array(df2['price'])
    x = data[:,0]
    y = data[:,1]
    x_test = testdata[:,0][0]
    y_test = testdata[:,1][0]
    z_real = np.array(test['price'])[0]
    display_pca_data(x,y,z)
    z_pred_scipy = scipy_fit_pred(x,y,z,x_test,y_test)
    z_pred_sklearn = sklearn_fit_pred(x,y,z,x_test,y_test)
    z_pred_statsmodel = statsmodel_fit_pred(x,y,z,x_test,y_test)
    z_pred_statsmodel_gamma = statsmodel_fit_pred_gamma(x,y,z,x_test,y_test)
    z_pred_statsmodel_reg1 = statsmodel_fit_pred_regular1(x,y,z,x_test,y_test)
    z_pred_statsmodel_reg2 = statsmodel_fit_pred_regular2(x,y,z,x_test,y_test)
    models = [z_pred_scipy, z_pred_sklearn, z_pred_statsmodel, z_pred_statsmodel_gamma, z_pred_statsmodel_reg1, z_pred_statsmodel_reg2]
    r2_scores = [r2_score(z, z_pred_scipy[1]), r2_score(z, z_pred_sklearn[1]), 
                 r2_score(z, z_pred_statsmodel[1]), r2_score(z, z_pred_statsmodel_gamma[1]), 
                 r2_score(z, z_pred_statsmodel_reg1[1]), r2_score(z, z_pred_statsmodel_reg1[1])]
    
    (xs,ys,zs) = models[np.argmax(r2_scores)][2]
    fig = plt.figure(figsize = (10,10))
    ax = Axes3D(fig)
    ax.plot_surface(xs, ys, zs, color ='purple', alpha = 0.4)
    ax.scatter(x, y, z, c = z, cmap='rainbow')
    ax.set_xlabel('X data')
    ax.set_ylabel('Y data')
    ax.set_zlabel('Z data')
    ax.view_init(30, 120)
    plt.show() 
    return z_real, models[np.argmax(r2_scores)][0], r2_scores[np.argmax(r2_scores)], np.argmax(r2_scores)

def display_pca_data(x,y,z): # This displays the data after pca from several angles
    fig = plt.figure(figsize = (14,7))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')

    ax.scatter3D(x, y, z, c=z, cmap='rainbow');
    ax.set_xlabel('X data')
    ax.set_ylabel('Y data')
    ax.set_zlabel('Z data')
    ax1.view_init(30, 0)

    ax1.scatter3D(x, y, z, c=z, cmap='rainbow');
    ax1.set_xlabel('X data')
    ax1.set_ylabel('Y data')
    ax1.set_zlabel('Z data')
    ax1.view_init(30, 60)

    fig2 = plt.figure(figsize = (14,7))
    ax = fig2.add_subplot(1, 2, 1, projection='3d')
    ax1 = fig2.add_subplot(1, 2, 2, projection='3d')

    ax.scatter3D(x, y, z, c=z, cmap='rainbow');
    ax.set_xlabel('X data')
    ax.set_ylabel('Y data')
    ax.set_zlabel('Z data')
    ax.view_init(30, 120)

    ax1.scatter3D(x, y, z, c=z, cmap='rainbow');
    ax1.set_xlabel('X data')
    ax1.set_ylabel('Y data')
    ax1.set_zlabel('Z data')
    ax1.view_init(30, 180)
    
def scipy_fit_pred(x,y,z,x_test,y_test): #linear model using scipy
  def function(data, c, d, e, f , g, h, i, j, k, l,m,n,o,p,q):
      x = data[0]
      y = data[1]
      return  c*x**3 + d*y**3 + e*y**2*x + f*y*x**2 + g*x**2 + h*y**2 + i*x*y + j*x + k*y + l + m*x**4 + n*y**4 + o*x**3*y + p*x*y**3 + q*x**2*y**2
  x1 = x
  y1 = y
  z1 = z
  parameters1, covariance1 = curve_fit(function, [x1, y1], z1)
  model_x1 = np.linspace(min(x1), max(x1), 100)
  model_y1 = np.linspace(min(y1), max(y1), 100)
  X1, Y1 = np.meshgrid(model_x1, model_y1)
  Z1 = function(np.array([X1, Y1]), *parameters1)
  z_fit = function([x,y], *parameters1)
  z_pred = function(np.array([x_test, y_test]), *parameters1)
 
  return z_pred, z_fit, (X1,Y1,Z1)   


def sklearn_fit_pred(x,y,z,x_test,y_test): #linear model using sklearn
    x1 = x
    y1 = y
    z1 = z


    def design(x,y):
        des = np.column_stack([(x**3).reshape(-1,1), (x**4).reshape(-1,1), (y**4).reshape(-1,1), (x**3*y).reshape(-1,1), 
                              (y**3).reshape(-1,1), (x**2*y).reshape(-1,1), (y**3*x).reshape(-1,1), (y**2*x**2).reshape(-1,1), 
                              (y**2*x).reshape(-1,1), (y**2).reshape(-1,1), (x**2).reshape(-1,1),
                              (x*y).reshape(-1,1), (x).reshape(-1,1),(y).reshape(-1,1), (np.ones(len(x.reshape(-1,1)))).reshape(-1,1)])
        return  des

    model1 = sklearn.linear_model.LinearRegression( fit_intercept=False)
    model1.fit(design(x1,y1), z1)
    model_x1 = np.linspace(min(x1), max(x1), 100)
    model_y1 = np.linspace(min(y1), max(y1), 100)
    X1, Y1 = np.meshgrid(model_x1, model_y1)
    Z1 = model1.predict(design(X1,Y1))
    Z1= np.reshape(Z1, (len(X1), len(X1)))
    z_fit = model1.predict(design(x1,y1))
    z_pred = model1.predict(design(x_test,y_test))

    return z_pred[0], z_fit, (X1,Y1,Z1)

  
def statsmodel_fit_pred(x,y,z,x_test,y_test): #poisson model using statsmodels

    x1 = x
    y1 = y
    z1 = z

    def design(x,y):
        des = np.column_stack([(x**3).reshape(-1,1), (x**4).reshape(-1,1), (y**4).reshape(-1,1), (x**3*y).reshape(-1,1), 
                              (y**3).reshape(-1,1), (x**2*y).reshape(-1,1), (y**3*x).reshape(-1,1), (y**2*x**2).reshape(-1,1), 
                              (y**2*x).reshape(-1,1), (y**2).reshape(-1,1), (x**2).reshape(-1,1),
                              (x*y).reshape(-1,1), (x).reshape(-1,1),(y).reshape(-1,1), (np.ones(len(x.reshape(-1,1)))).reshape(-1,1)])
        return  des

    m_linb1 = sm.GLM(z1,design(x1,y1), family=sm.families.Poisson())

    rb1 = m_linb1.fit()
    model_x1 = np.linspace(min(x1), max(x1), 100)
    model_y1 = np.linspace(min(y1), max(y1), 100)
    X1, Y1 = np.meshgrid(model_x1, model_y1)
    Z1 = rb1.predict(design(X1,Y1))
    Z1= np.reshape(Z1, (len(X1), len(X1)))

    z_fit = rb1.predict(design(x1,y1))
    z_pred = rb1.predict(design(x_test,y_test))

    return z_pred[0], z_fit, (X1,Y1,Z1)

def statsmodel_fit_pred_gamma(x,y,z,x_test,y_test): #gamma model using statsmodels

    x1 = x
    y1 = y
    z1 = z
    def design(x,y):
        des = np.column_stack([(x**3).reshape(-1,1), (x**4).reshape(-1,1), (y**4).reshape(-1,1), (x**3*y).reshape(-1,1), 
                              (y**3).reshape(-1,1), (x**2*y).reshape(-1,1), (y**3*x).reshape(-1,1), (y**2*x**2).reshape(-1,1), 
                              (y**2*x).reshape(-1,1), (y**2).reshape(-1,1), (x**2).reshape(-1,1),
                              (x*y).reshape(-1,1), (x).reshape(-1,1),(y).reshape(-1,1), (np.ones(len(x.reshape(-1,1)))).reshape(-1,1)])
        return  des

    m_linb1 = sm.GLM(z1,design(x1,y1), family=sm.families.Gamma())
    rb1 = m_linb1.fit()
    model_x1 = np.linspace(min(x1), max(x1), 100)
    model_y1 = np.linspace(min(y1), max(y1), 100)
    X1, Y1 = np.meshgrid(model_x1, model_y1)
    Z1 = rb1.predict(design(X1,Y1))
    Z1= np.reshape(Z1, (len(X1), len(X1)))

    z_fit = rb1.predict(design(x1,y1))
    z_pred = rb1.predict(design(x_test,y_test))

    return z_pred[0], z_fit, (X1,Y1,Z1)

def statsmodel_fit_pred_regular1(x,y,z,x_test,y_test): #linear model using statsmodels lasso regression

    x1 = x
    y1 = y
    z1 = z

    def design(x,y):
        des = np.column_stack([(x**3).reshape(-1,1), (x**4).reshape(-1,1), (y**4).reshape(-1,1), (x**3*y).reshape(-1,1), 
                              (y**3).reshape(-1,1), (x**2*y).reshape(-1,1), (y**3*x).reshape(-1,1), (y**2*x**2).reshape(-1,1), 
                              (y**2*x).reshape(-1,1), (y**2).reshape(-1,1), (x**2).reshape(-1,1),
                              (x*y).reshape(-1,1), (x).reshape(-1,1),(y).reshape(-1,1), (np.ones(len(x.reshape(-1,1)))).reshape(-1,1)])
        return  des

    m_linb1 = sm.GLM(z1,design(x1,y1), family=sm.families.Gaussian())
    rb1 = m_linb1.fit_regularized(alpha=0.05,L1_wt=0.2)
    model_x1 = np.linspace(min(x1), max(x1), 100)
    model_y1 = np.linspace(min(y1), max(y1), 100)
    X1, Y1 = np.meshgrid(model_x1, model_y1)
    Z1 = rb1.predict(design(X1,Y1))
    Z1= np.reshape(Z1, (len(X1), len(X1)))
    z_fit = rb1.predict(design(x1,y1))
    z_pred = rb1.predict(design(x_test,y_test))

    return z_pred[0], z_fit, (X1,Y1,Z1)

def statsmodel_fit_pred_regular2(x,y,z,x_test,y_test): #linear model using statsmodels ridge regression

    x1 = x
    y1 = y
    z1 = z

    def design(x,y):
        des = np.column_stack([(x**3).reshape(-1,1), (x**4).reshape(-1,1), (y**4).reshape(-1,1), (x**3*y).reshape(-1,1), 
                              (y**3).reshape(-1,1), (x**2*y).reshape(-1,1), (y**3*x).reshape(-1,1), (y**2*x**2).reshape(-1,1), 
                              (y**2*x).reshape(-1,1), (y**2).reshape(-1,1), (x**2).reshape(-1,1),
                              (x*y).reshape(-1,1), (x).reshape(-1,1),(y).reshape(-1,1), (np.ones(len(x.reshape(-1,1)))).reshape(-1,1)])
        return  des

    m_linb1 = sm.GLM(z1,design(x1,y1), family=sm.families.Gaussian())
    rb1 = m_linb1.fit_regularized(alpha=0.05,L1_wt=0.7)
    model_x1 = np.linspace(min(x1), max(x1), 100)
    model_y1 = np.linspace(min(y1), max(y1), 100)
    X1, Y1 = np.meshgrid(model_x1, model_y1)
    Z1 = rb1.predict(design(X1,Y1))
    Z1= np.reshape(Z1, (len(X1), len(X1)))
    z_fit = rb1.predict(design(x1,y1))
    z_pred = rb1.predict(design(x_test,y_test))

    return z_pred[0], z_fit, (X1,Y1,Z1)

def predict_z_models_allvecs(df2, test, components): #this function does the actual prediction and returns the best model for all the vectors
    data1, testdata1 = prin_comp(components, df2, test, 'vector_distance')
    data2, testdata2 = prin_comp(components, df2, test, 'vector_count')
    data3, testdata3 = prin_comp(components, df2, test, 'vector_distance_cat')
    data4, testdata4 = prin_comp(components, df2, test, 'vector_count_cat')
    z = np.array(df2['price'])
    xs = [data1[:,0], data2[:,0], data3[:,0], data4[:,0]]
    ys = [data1[:,1], data2[:,1], data3[:,1], data4[:,1]]

    x_tests = [testdata1[:,0][0], testdata2[:,0][0] , testdata3[:,0][0] , testdata4[:,0][0]]
    y_tests = [testdata1[:,1][0], testdata2[:,1][0] , testdata3[:,1][0] , testdata4[:,1][0]]
    z_real = np.array(test['price'])[0]

    def apply_fits(x,y,z,x_test,y_test):
      z1= scipy_fit_pred(x,y,z,x_test,y_test)
      z2 =sklearn_fit_pred(x,y,z,x_test,y_test)
      z3 = statsmodel_fit_pred(x,y,z,x_test,y_test)
      z4 =statsmodel_fit_pred_gamma(x,y,z,x_test,y_test)
      z5 = statsmodel_fit_pred_regular1(x,y,z,x_test,y_test)
      z6 = statsmodel_fit_pred_regular2(x,y,z,x_test,y_test)
      return [z1,z2,z3,z4,z5,z6]

    results =[]
    results = results + apply_fits(xs[0],ys[0],z,x_tests[0],y_tests[0])
    results = results + apply_fits(xs[1],ys[1],z,x_tests[1],y_tests[1])  
    results = results + apply_fits(xs[2],ys[2],z,x_tests[2],y_tests[2]) 
    results = results + apply_fits(xs[3],ys[3],z,x_tests[3],y_tests[3])
    

    def r2s(val):
      return r2_score(z, val[1])

    scores = list(map(r2s, results))
    val = np.argmax(scores)

    display_pca_data(xs[val//6],ys[val//6],z)
    (Xs,Ys,Zs) = results[val][2]

    fig = plt.figure(figsize = (10,10))
    ax = Axes3D(fig)
    ax.plot_surface(Xs,Ys,Zs, color ='purple', alpha = 0.4)
    ax.scatter(xs[val//6], ys[val//6], z, c = z, cmap='rainbow')
    ax.set_xlabel('X data')
    ax.set_ylabel('Y data')
    ax.set_zlabel('Z data')
    ax.view_init(30, 120)
    plt.show() 
    return z_real, results[val][0], scores[val], val
