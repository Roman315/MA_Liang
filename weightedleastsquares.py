import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from scipy.sparse import dia_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve, norm
from numpy.random import rand
from scipy.linalg import solve_banded


def weightedlsq(df, threshold):
    
    df_long = df.unstack().reset_index() # from 768x1024 zu 786432x3 Matrix
    # df_long.columns = ["X","Y","Z"] # rename columns 
    df_long = df_long.to_numpy() # pandas to numpy
    
    # divide df_long in vectors
    x1 = df_long[:,0] 
    x2 = df_long[:,1]
    y = df_long[:,2] # height-data
    
    # yi = beta0 + beta1*x1 + beta2*x2
    # y: Höhenwerte, X: [1 x1 x2], beta: (beta0 beta1 beta2)
    # y = X*beta
    
    X = np.vstack([np.ones(len(x1)), x1, x2]).T # X = [1 x1 x2]
    X = np.matrix(X) 
    y_org = np.matrix(y).T # height-data
    y = y_org
    
    #w = 1 # start-weight
    #W = np.diag(np.full(len(x1),w)) # vector on diag. Matrix
    e = np.ones(len(x1)) 
    
    #beta = (X.T * W * X).I * X.T * W * y_org # compute Least Square Solution
    #print(beta)

    #Test
    W = lil_matrix((len(x1), len(x1)))
    W.setdiag(np.ones(len(x1)))
    
    #W = dia_matrix((w, 0))
    W = W.tocsr()
    beta = spsolve((X.T * X), X.T * y_org)
    beta = np.matrix(beta).T

    #error = y_org-X*beta # compute error
    #df_error = np.reshape(error, (df.shape[1], df.shape[0])).T

    e_diff = 100000
    count = 0
    while e_diff > threshold: # iteration until change of the error is smaller than threshold
    #while count < 5:
        e_old = e # save old error
        
        y = X*beta    # COMPUTE y = X*beta
        e = y_org - y # calculate error
        
        w = np.exp(-abs(e))  # UPDATE weights 
        #w = np.exp(-(np.square(e))) # UPDATE w
        #w = 1/(abs(e)) # UPDATE w
        
        #delta = np.ones(len(X))
        #delta.fill(0.0001) # to avoid division by zero
        #delta = np.matrix(delta).T
        #w = 1/np.maximum(delta, abs(e)) # UPDATE w
        
        #ALT
        #W = np.diag(w.A1) # vector on diag. Matrix
        #W = np.matrix(W)
        #beta = (X.T * W * X).I * X.T * W * y_org # UPDATE weighted Least Square Solution
        
        W = dia_matrix((w.T,0),(len(x1), len(x1)))
        #W.setdiag(w)
        #beta = solve_banded((X.T * W * X), X.T * W * y_org)
        
        #Test
        
        #W = lil_matrix((len(x1), len(x1)))
        #W.setdiag(w)
        W = W.tocsr()
        
        X = csr_matrix(X)
        
        A = csr_matrix(X.T * W * X)
        
        beta = spsolve(A, X.T * W * y_org)
        #beta.toarray()
        beta = np.matrix(beta).T
        
        e_diff = abs((sum(abs(e_old))) - (sum(abs(e)))) # difference of previous error to new error
        #e_diff = abs(sum(e_old - e))
        #print('ediff=',e_diff)
        count = count + 1
        
    df_error = np.reshape(e, (df.shape[1], df.shape[0])).T # errors in dataframe
    df_error = pd.DataFrame(df_error) # numpy to pandas
    df_weights = np.reshape(w, (df.shape[1], df.shape[0])).T # weights in dataframe
    df_weights = pd.DataFrame(df_weights) # numpy to pandas
    
    print('Iterationen Least Square:', count)
    
    return X, y, y_org, beta, W, df_error, e, w, df_weights