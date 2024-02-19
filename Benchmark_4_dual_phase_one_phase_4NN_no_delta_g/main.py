import subprocess
import sys

import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '-1' 






#tf.print("PYTHON VERSION: ",sys.version)
# Install pyDOE using pip
subprocess.call(['pip', 'install', 'pyDOE'])

#!pip install pyDOE

import datetime, os

 
#0 (default) shows all, 1 to filter out INFO logs, 2 to additionally filter out WARNING logs, and 3 to additionally filter out ERROR logs
import scipy.optimize
import scipy.io
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend (non-interactive)
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import time
import psutil # memory usage
from pyDOE import lhs         #Latin Hypercube Sampling

import codecs, json  # save weights
import math
import glob
#from numba import jit
# generates same random numbers each time


import tensorflow as tf



np.random.seed(1234)

import random
import datetime
import shutil

import random
import scipy.io as sio
from importlib import reload
#from sklearn.preprocessing import MinMaxScaler



import PINN  # python files (classes)
import pre_post
from pre_post import *
from PINN import *







def read_inputs_from_file(file_path):
    variables = {}

    # Save the variables to the dictionary
    with open(file_path, "r") as file:
        for line in file:
            key, value = line.strip().split("=")
            key = key.strip()
            value = value.strip()

            try:
                # Check if the value can be converted to an integer
                int_value = int(value)
                variables[key] = int_value
            except ValueError:
                try:
                    # Check if the value can be converted to a floating-point number
                    float_value = float(value)
                    variables[key] = float_value
                except ValueError:
                    # Otherwise, it is a string value
                    variables[key] = value

    return variables

#@jit(nopython=True)
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

##################################################
##################################################
##################################################
if __name__ == '__main__':     ###################
##################################################
##################################################
##################################################    
    #inputs = read_inputs_from_file("Input.txt")

    # Grid parameters
    Nx=65
    Ny=65
    Nt=150 # 500 if los_f
    dx = 1e-6 
    dy = dx 
    # physical parameters
    dt =1e-3
    sigma=0.24
    mu=1e-8 # 1e-8 
    delta_g= 0 
    eta=7*dx




    # Define the domain bounds
    lb = np.array([0, 0,0])
    ub = np.array([Nx*dx, Ny*dy,Nt*dt]) 

    x = np.linspace(lb[0], ub[0], Nx)
    y = np.linspace(lb[1], ub[1], Ny)
    t= np.linspace(lb[2], ub[2], Nt) 
        

    num_phases=1

    num_phases=1
    loc_index_0 = 0
    loc_index_1 = 1
    loc_index_2 = 2
    loc_index_3 = 3
    all_phases_indexes= [loc_index_0, loc_index_1, loc_index_2, loc_index_3]
    
    N_batches=4 # base of the pyramid
    min_batch_numbers =4 # upper surface of the pyramid
    # Training batch 
    Nbr_f_pts_max_per_batch= 50 #1000
    Nbr_f_pts_min_per_batch= 40 #100
    N_ini_max_per_batch=90  #50   
    N_ini_min_per_batch=80 #4      
    fraction_ones_per_int_pts=0.
    fraction_zeros_per_int_pts=0.
    coef_increase_points_f=2 # or decrease
    coef_increase_points_ic=2 # or decrease
    

    
    num_train_intervals=Nt
    # Define  Collocations, IC and BC points and Domain bounds
    N_ini =N_ini_max_per_batch*N_batches *num_train_intervals # Total number of data points for 'phi': IC
    N_f = Nbr_f_pts_max_per_batch * N_batches *num_train_intervals    # 100000 Total number of collocation points : domain
    N_b=N_batches*Nt   # Total number of data points for boundary BC
        

    f_values = [0, 0.33, 0.67, 1]

    # Generate all combinations of x, y, t, and f
    X, Y, T, F = np.meshgrid(np.linspace(lb[0], ub[0], Nx),
                            np.linspace(lb[1], ub[1], Ny),
                            np.linspace(lb[2], ub[2], Nt),
                            f_values,
                            indexing='ij')

    # Reshape the arrays to create the test matrix
    X_phi_test=np.column_stack((X.flatten(), Y.flatten(), T.flatten()))
    np.savez('X_phi_test.npz', X_phi_test=X_phi_test)
    
    #X_phi_test = np.load('X_phi_test.npz')['arr_0']
    data = np.load('X_phi_test.npz')
    X_phi_test= data['X_phi_test']

    tb = np.linspace(start=lb[2], stop=ub[2], num=N_b, endpoint=True)
    tb = np.expand_dims(tb, axis=1)
    
    # set the saving paths and erase older results
    global pathOutput
    pathOutput = os.path.join(os.getcwd(),'save_figs')
    if not os.path.isdir(pathOutput):
        os.mkdir(pathOutput)
    global pathInput
    pathInput = os.path.join(os.getcwd(),'Initialization')
    if not os.path.isdir(pathInput):
        os.mkdir(pathInput)
    # to store the weights for each time interval 
    path_weights= os.path.join(os.getcwd(),'weights')
    if not os.path.isdir(path_weights):
        os.mkdir(path_weights)

    # load PrePost class
    Pre_Post=PrePost(X=X,T=None, lb=lb, ub=ub, Nx=Nx,Ny=Ny,dx=dx,dy=dy,x=x,y=y, eta=eta,\
            phi_true=None)

    # set the save paths and erase older results
    Pre_Post.EraseFile(path=pathOutput)
    Pre_Post.EraseFile(path=path_weights)
    Pre_Post.EraseFile(path=os.path.join(os.getcwd(),'test_IC'))
    #Pre_Post.EraseFile(path=pathInput) # Initialization 


    # Initialize phases
    # get radius and coordinates
    R0, X_center, Y_center,Z_center =\
        generate_circles(mean_r=25*dx,num_circles=1, std=0, Nx=Nx, Ny=Ny,Nz=100)
    X_center=np.array([Nx*dx/2])
    Y_center=np.array([Ny*dx/2])
    
    phi_0, X_ini_all=Pre_Post.init_micro_cir(R0,X_center,Y_center, Z_center,N_ini,Nx,Ny,x,y,lb,ub) 
    phi_0=phi_0.reshape(1, Nx, Ny)
    X_ini_all=X_ini_all.reshape(1, Nx, Ny)
    # Save into a dictionary
    Initilization_Data = {'phi_0': phi_0}
    np.savez('Initilization_Data_dual.npz', **Initilization_Data)

    # Dual
    loaded_data = np.load('Initilization_Data_dual.npz')
    all_phases = loaded_data['phi_0']
    all_interfaces = np.zeros_like(all_phases)
    all_flags_matrix=np.zeros_like(all_phases)
    all_phases_indexes=np.zeros_like(all_phases)
    phases_indexes=all_phases_indexes
    """
    tf.print("phases_indexes shape:", phases_indexes.shape)
    tf.print("all_flags_matrix shape:", all_flags_matrix.shape)
    tf.print("all_phases shape:", all_phases.shape)
    tf.print("all_interfaces shape:", all_interfaces.shape)
    """
    # plot the initial micro
    #Pre_Post.plot_init(all_phases,all_phases,Nx,Ny,path=pathOutput)


    # Time characteristic
    T=eta**2 /(sigma*mu)
    # Length characteristic : eta
    # scaled arrays
    x=x/eta
    y=y/eta
    t=t/T 
    dt =dt /T 
    dx=dx/eta
    dy=dy/eta
    delta_g=eta*delta_g
    lb[0], ub[0]=lb[0]/eta, ub[0]/eta
    lb[1], ub[1]= lb[1]/eta, ub[1]/eta
    lb[2], ub[2]= lb[2]/T, ub[2]/T
    #ub=ub/eta

    # get the training data
    X_f_train, X_ini_train_all,X_lb_train,X_ub_train,X_rtb_train,X_ltb_train,phi_ini_all =Pre_Post.set_training_data(x,y,N_ini,\
        all_phases, all_interfaces,all_flags_matrix,N_f,tb,lb,ub,path=pathOutput)
    
    # Plot Collocation_IC_BC points
    #Pre_Post.plot_Collocation_IC_BC(Nx,Ny,x,y,X_ini_train,X_f_train,X_lb_train,X_ub_train,\

    # load PINN class
    ######################################################################## 
    # Build PINN 
    layers = np.array([3,128,128,128,128,128,128,1])  # Network
    PINN_ = Sequentialmodel(layers=layers, X_f_train=X_f_train, X_ini_train=X_ini_train_all,\
                            phases_ini_indexes=phases_indexes,all_ini_flags_matrix=all_flags_matrix,\
                            Phi_ini=all_phases,phi_ini_train=phi_ini_all, N_ini=N_ini,X_phi_test=X_phi_test,\
                            X_ini_train_all=X_ini_train_all, phi_ini_train_all=phi_ini_all,\
                                all_interfaces=all_interfaces,\
                            X_lb_train=X_lb_train, X_ub_train=X_ub_train,\
                            X_ltb_train=X_ltb_train, X_rtb_train=X_rtb_train,\
                            X=X,Y=Y,T=T,x=x,y=y,lb=lb, ub=ub, mu=mu, sigma=sigma, delta_g=delta_g,\
                            eta=eta,Nx=Nx,Ny=Ny,Nt=Nt,phi_sol=None,pinns =None,num_phases=num_phases,
                            N_batches=N_batches,\
                            min_batch_numbers = min_batch_numbers,\
                            Nbr_f_pts_max_per_batch=Nbr_f_pts_max_per_batch,\
                            Nbr_f_pts_min_per_batch=Nbr_f_pts_min_per_batch,\
                            N_ini_max_per_batch=N_ini_max_per_batch,\
                            N_ini_min_per_batch=N_ini_min_per_batch)
    ########################################################################   
    # New dir for pinns (the workers) and initialization
    path_weights_all_pinns= os.path.join(os.getcwd(),'weights_all_workers_pinns')
    if not os.path.isdir(path_weights_all_pinns):
        os.mkdir(path_weights_all_pinns)
    pinns=PINN_.Initialize_pinns(path_weights_all_pinns)  

    ########################################################################      
    # transfer learning from already trained model
    set_weights_PINN=0
    if set_weights_PINN==1:
        PINN_.set_weights_Master_PINN()
        pinns=PINN_.Initialize_pinns(path_weights_all_pinns) 
        PINN_.test_IC(N_batches,"test_IC")
    ########################################################################      

    Pre_Post.EraseFile(path=path_weights_all_pinns)  
    ########################################################################      

    # train the Master model
    global Nfeval
    Nfeval = 1 #(c.f. PINN.py -- scipy optimizer)
    start_time = time.time() 
    ########################################################################  

    list_loss= PINN_.train(epochs=50000,batch_size_max=1000,thresh=5e-5,epoch_scipy_opt=50,epoch_print=10,\
                            epoch_resample=1000000,initial_check=True,save_reg_int=500000,\
                            num_train_intervals=num_train_intervals,\
                            discrete_resolv=True,\
                                fraction_ones_per_int_pts=fraction_ones_per_int_pts,\
                                fraction_zeros_per_int_pts=fraction_zeros_per_int_pts,\
                                coef_increase_points_f=coef_increase_points_f,\
                                coef_increase_points_ic=coef_increase_points_ic,\
                            path=pathOutput,path_weights_all_pinns=path_weights_all_pinns,\
                                save_weights_pinns=False,communicate_pinns=True,\
                                    change_pinn_candidate=False,Thresh_Master=1e-2,\
                                    optimize_master=False, transfer_learning=True,\
                                    denoising_loss=False, loss_sum_constraint=False)  

    elapsed = time.time() - start_time  
    tf.print("Training time : " + (str(datetime.timedelta(seconds=elapsed))) )

