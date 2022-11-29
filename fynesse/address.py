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

def hello_world():
    print("This is my first pip package!")

def predict_data(latitude, longitude, date, property_type, conn):
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
    df['vector_distance_cat'] = assess.vec_app(df,0.02, pois, assess.get_vector_inv_cat)
    df['vector_distance'] = assess.vec_app(df,0.02, pois, assess.get_vector_distance)
    df['vector_count'] = assess.vec_app(df,0.02, pois, assess.get_vector_count)
    df['vector_count_cat'] = assess.vec_app(df, 0.02,pois, assess.get_vector_count_cat)
    test = df[(df['lattitude'] == latitude) & (df['longitude']	== longitude ) & (df['date_of_transfer'] == datedate)]
    df = df[(df['lattitude'] != latitude) | (df['longitude']	!= longitude ) | (df['date_of_transfer'] != datedate)]

    return df, test

  
def eigen_view(df2, column='vector_distance_cat'):
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
  
def prin_comp(components, df2, test, column='vector_distance_cat' ):
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
  
def predict_z(df2, test, components,column):
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
    return z_real, z_pred_scipy, z_pred_sklearn, z_pred_statsmodel  

def display_pca_data(x,y,z):
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
    
def scipy_fit_pred(x,y,z,x_test,y_test):
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
  z_pred = function(np.array([x_test, y_test]), *parameters1)

  fig = plt.figure(figsize = (10,10))

  ax = Axes3D(fig)

  ax.plot_surface(X1, Y1, Z1, color ='purple', alpha = 0.4)
  ax.scatter(x1, y1, z1, c =z1, cmap='rainbow')

  ax.set_xlabel('X data')
  ax.set_ylabel('Y data')
  ax.set_zlabel('Z data')
  ax.view_init(30, 120)
  plt.show()   
  return z_pred    


def sklearn_fit_pred(x,y,z,x_test,y_test):
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

    z_pred = model1.predict(design(x_test,y_test))

    fig = plt.figure(figsize = (10,10))
    # setup 3d object
    ax = Axes3D(fig)
    # plot surface
    ax.plot_surface(X1, Y1, Z1, color ='purple', alpha = 0.4)
    # plot input data
    ax.scatter(x1, y1, z1, c =z1, cmap='rainbow')
    # set plot descriptions
    ax.set_xlabel('X data')
    ax.set_ylabel('Y data')
    ax.set_zlabel('Z data')
    ax.view_init(30, 180)
    plt.show() 
    return z_pred
  
def statsmodel_fit_pred(x,y,z,x_test,y_test):

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
    z_pred = rb1.predict(design(x_test,y_test))


    fig = plt.figure(figsize = (10,10))
    # setup 3d object
    ax = Axes3D(fig)
    # plot surface
    ax.plot_surface(X1, Y1, Z1, color ='purple', alpha = 0.4)
    # plot input data
    ax.scatter(x1, y1, z1, c =z1, cmap='rainbow')
    # set plot descriptions
    ax.set_xlabel('X data')
    ax.set_ylabel('Y data')
    ax.set_zlabel('Z data')
    ax.view_init(30,120)
    plt.show() 
    return z_pred
