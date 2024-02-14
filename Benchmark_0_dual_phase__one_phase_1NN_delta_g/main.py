import subprocess
import sys


#!/home/selfetni/anaconda3/bin/python3.9.19
#print("PYTHON VERSION: ",sys.version)
# Install pyDOE using pip
subprocess.call(['pip', 'install', 'pyDOE'])

#!pip install pyDOE

import tensorflow as tf
import datetime, os
#hide tf logs 
os.environ['TF_CPPclea_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'} 
#0 (default) shows all, 1 to filter out INFO logs, 2 to additionally filter out WARNING logs, and 3 to additionally filter out ERROR logs
import scipy.optimize
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import time
import psutil # memory usage
from pyDOE import lhs         #Latin Hypercube Sampling
import seaborn as sns 
import codecs, json  # save weights
import math
import glob
from numba import jit
# generates same random numbers each time
np.random.seed(1234)
tf.random.set_seed(1234)
import random
import datetime
import shutil
print("TensorFlow version: {}".format(tf.__version__))
import random
import scipy.io as sio
from importlib import reload
import PINN  # python files (classes)
import pre_post
from pre_post import *
from PINN import *

#generate_circles without overlap
@jit(nopython=True)
def generate_circles(mean_r, num_circles, std, Nx, Ny, Nz):
    # Initialize the arrays for the radii and centers of the circles
    R0 = np.zeros(num_circles)
    X_center = np.zeros(num_circles)
    Y_center = np.zeros(num_circles)
    Z_center = np.zeros(num_circles)

    # Generate the first circle randomly
    R0[0] = np.random.normal(loc=mean_r, scale=std)
    X_center[0] = np.random.randint(R0[0], Nx-R0[0])
    Y_center[0] = np.random.randint(R0[0], Ny-R0[0])
    Z_center[0] = np.random.randint(R0[0], Nz-R0[0])

    # Loop through the remaining circles and generate them one at a time
    for i in range(1, num_circles):
        # Flag to indicate whether the new circle overlaps with any existing circles
        overlaps = True
        while overlaps:
            # Generate the radius and center of the new circle randomly
            R0[i] = np.random.normal(loc=mean_r, scale=std)
            X_center[i] = np.random.randint(R0[i], Nx-R0[i])
            Y_center[i] = np.random.randint(R0[i], Ny-R0[i])
            Z_center[i] = np.random.randint(R0[i], Nz-R0[i])

            # Check the new circle against the existing circles
            overlaps = False
            for j in range(i):
                if np.sqrt((X_center[i]-X_center[j])**2 + (Y_center[i]-Y_center[j])**2 ) < (R0[i]+R0[j]): #+ (Z_center[i]-Z_center[j])**2
                    overlaps = True
                    break
    
    return R0, X_center, Y_center, Z_center

if __name__ == '__main__':
    # Grid parameters
    Nx=64
    Ny=64
    Nt=100
    eta=0.075  
    # Define the domain bounds
    lb = np.array([0, 0,0])
    ub = np.array([1, 1,1]) #np.array([Nx, Ny,Nt])
    
    # Phyiscal paramters
    v_n=0.5
    sigma=1
    mu=1e-3
    delta_g=-100
    
    # Numerical parameters 
    N_batches=36 # Total number of data points for 'phi': boundary 
    Nbr_pts_max_per_batch=12500  # maximum number  Collocation points per batch for Scipy optimizer 
    scipy_min_f_pts_per_batch = 500 # minimum number of Collocation points per batch for Scipy optimizer 
    max_ic_scipy_pts=75 #maximum number of IC points per batch for Scipy optimizer 
    N_ini_min_per_batch=10
    
    scipy_min_f_pts_per_batch_thresh =0.05  # to delete
    ic_scipy_thresh=0.05 # 
    
    num_train_intervals=100
    # Define  Collocations, IC and BC points and Domain bounds
    N_ini =N_batches *num_train_intervals # Total number of data points for 'phi': IC
    N_f = Nbr_pts_max_per_batch * N_batches *num_train_intervals    # 100000 Total number of collocation points : domain
    N_b=500   # Total number of data points for boundary BC
        
    # Total number of data points for 'phi': boundary BC
    # get radius and coordinates
    R0, X_center, Y_center,Z_center =\
          generate_circles(mean_r=0.3,num_circles=1, std=0, Nx=Nx, Ny=Ny,Nz=100)
    X_center, Y_center=[0.5],[0.5] # single grain
    x = np.linspace(lb[0], ub[0], Nx),
    y = np.unique(np.linspace(lb[1], ub[1], Ny))
    t = np.unique(np.linspace(lb[2], ub[2], Nt))
    x=np.expand_dims(x, axis=1)
    y=np.expand_dims(y, axis=1)
    t=np.expand_dims(t, axis=1)
    X, Y,T= np.meshgrid(x,y,t)
    tb = np.linspace(start=lb[2], stop=ub[2], num=N_b, endpoint=True)
    tb = np.expand_dims(tb, axis=1)
    
    # set the saving paths and erase older results
    global pathOutput
    pathOutput = os.path.join(os.getcwd(),'save_figs')
    if not os.path.isdir(pathOutput):
        os.mkdir(pathOutput)
    # to store the weights for each time interval 
    path_weights= os.path.join(os.getcwd(),'weights')
    if not os.path.isdir(path_weights):
        os.mkdir(path_weights)

    # load PrePost class
    reload(pre_post) # for re-execution after modif
    from pre_post import *
    Pre_Post=PrePost( X=X,T=None, lb=lb, ub=ub, Nx=Nx,Ny=Ny,x=x,y=y, eta=eta,\
                      phi_true=None,R0=R0)

    # set the save paths and erase older results
    Pre_Post.EraseFile(path=pathOutput)
    Pre_Post.EraseFile(path=path_weights)
 
    # get phi_0
    phi_0, X_ini_all=Pre_Post.init_micro_cir(X_center,Y_center, Z_center,N_ini,Nx,Ny,x,y,lb,ub) 
    
    # plot the true solution
    #Pre_Post.plot_exact(path=pathOutput)

    # plot the initial micro
    #Pre_Post.plot_init(X_ini_all,phi_0,Nx,Ny,path=pathOutput)
    
    # get the training data
    X_f_train, X_ini_train,X_lb_train,X_ub_train,X_rtb_train,X_ltb_train,\
        phi_ini_train, X_ini_train_all, phi_ini_train_all=Pre_Post.set_training_data(x,y,X_ini_all,N_ini,phi_0,N_f,tb,lb,ub)
 
    # Plot Collocation_IC_BC points
    #Pre_Post.plot_Collocation_IC_BC(Nx,Ny,x,y,X_ini_train,X_f_train,X_lb_train,X_ub_train,\
    #                                X_rtb_train,X_ltb_train,phi_0,phi_ini_train,path=pathOutput)
      
    # Testing spatio-temporal domain
    X_phi_test = np.hstack((X.flatten()[:,None],Y.flatten()[:,None], T.flatten()[:,None])) 
    #tf.print(X_phi_test)
 
    # load PINN class
    import PINN
    reload(PINN)  # mandatory to reload content at each re-call atfer modification
    from PINN import *
    
    # Build PINN 
    layers = np.array([3,128,128, 128,128,128,128,1])  # Network
    PINN_ = Sequentialmodel(layers=layers, X_f_train=X_f_train, X_ini_train=X_ini_train,\
                            X_lb_train=X_lb_train, X_ub_train=X_ub_train,\
                            X_ltb_train=X_ltb_train, X_rtb_train=X_rtb_train,\
                            phi_0=phi_0,phi_ini_train=phi_ini_train, N_ini=N_ini,X_u_test=X_phi_test,\
                            X_ini_train_all=X_ini_train_all, phi_ini_train_all=phi_ini_train_all,\
                            X=X,T=T,x=x,y=y,lb=lb, ub=ub, mu=mu, sigma=sigma, delta_g=delta_g,\
                                        R0=R0,X_center=X_center,Y_center=Y_center,eta=eta,Nx=Nx,Ny=Ny,Nt=Nt,phi_sol=None)
    
    
    # transfer learning from already trained model
    weights_files = glob.glob('get_weights/*.json')
    weights_files = sorted(weights_files)
    weights_file = weights_files[-1]
    t_min, t_max = weights_file.split('_')[2:4]
    with open(weights_file, 'r') as f:
        weights_loaded =json.load(f)['weights']
    weights_loaded=tf.cast(weights_loaded, dtype=tf.float64)
    PINN_.set_weights(weights_loaded) 
    

    # test the transfer of the learning
    PINN_.test_IC(pathOutput)
    
    Nfeval = 1  # global print variable
    start_time = time.time() 
    # train the model with Scipy L-BFGS optimizer  
                   
    list_loss= PINN_.train(epochs=50000,batch_size_max=1000,N_batches=N_batches,thresh= 5e-3,epoch_scipy_opt=1,epoch_print=50,\
                               epoch_resample=1,initial_check=True,save_reg_int=50,\
                               num_train_intervals=num_train_intervals,Nbr_pts_max_per_batch=Nbr_pts_max_per_batch,\
                               scipy_min_f_pts_per_batch=scipy_min_f_pts_per_batch,scipy_min_f_pts_per_batch_thresh=scipy_min_f_pts_per_batch_thresh,\
                                   max_ic_scipy_pts=max_ic_scipy_pts,N_ini_min_per_batch=N_ini_min_per_batch, ic_scipy_thresh=ic_scipy_thresh,\
                                   discrete_resolv=True,path=pathOutput)  

    elapsed = time.time() - start_time  
    print("Training time : " + (str(datetime.timedelta(seconds=elapsed))) ) 
