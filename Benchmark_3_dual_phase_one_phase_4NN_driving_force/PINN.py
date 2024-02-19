import tensorflow as tf
import datetime, os
#hide tf logs 

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
import sys
import time
import psutil
from pyDOE import lhs         #Latin Hypercube Sampling
import seaborn as sns 
import codecs, json
import math
#import numba
# generates same random numbers each time
np.random.seed(1234)
tf.random.set_seed(1234)
import random
import datetime
import shutil
import random
import glob 
import multiprocessing
import logging
import traceback
import gc
import copy
from multiprocessing import Pool, Lock
from scipy.spatial import cKDTree
#import jax
#import jax.numpy as jnp
#from jax.scipy.optimize import minimize
#import jax.numpy as jnp
#import jax.scipy as jsp

from importlib import reload
import pre_post
reload(pre_post) 
from pre_post import *

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)
    
class Sequentialmodel(tf.Module):
    ###############################################
    def __init__(self,layers, X_f_train, X_ini_train,\
                            phases_ini_indexes,all_ini_flags_matrix,\
                            Phi_ini,phi_ini_train, N_ini,X_phi_test,\
                            X_ini_train_all, phi_ini_train_all,\
                                all_interfaces,\
                            X_lb_train, X_ub_train,\
                            X_ltb_train, X_rtb_train,\
                            X,Y,T,x,y,lb, ub, mu, sigma, delta_g,\
                            eta,Nx,Ny,Nt,phi_sol,pinns,num_phases,\
                            N_batches,\
                            min_batch_numbers,    \
                            Nbr_f_pts_max_per_batch,\
                            Nbr_f_pts_min_per_batch,\
                            N_ini_max_per_batch,\
                            N_ini_min_per_batch,name=None):
        super().__init__(name=name)
         
        self.X_f = X_f_train            
        self.X_ini = X_ini_train
        self.phi_ini = phi_ini_train
        self.X_lb = X_lb_train
        self.X_ub = X_ub_train
        self.X_ltb = X_ltb_train
        self.X_rtb = X_rtb_train

        self.X_f_sub_domain = X_f_train  # just for initialization, to be updated            
        self.X_ini_sub_domain = X_ini_train # just for initialization, to be updated 
        self.phi_ini_sub_domain = phi_ini_train # just for initialization, to be updated 
        self.X_lb_sub_domain = X_lb_train # just for initialization, to be updated 
        self.X_ub_sub_domain = X_ub_train # just for initialization, to be updated 
        self.X_ltb_sub_domain = X_ltb_train # just for initialization, to be updated 
        self.X_rtb_sub_domain = X_rtb_train # just for initialization, to be updated 
        
        self.X_f_sub_domain_scipy=X_f_train  # just for initializatio, to be updated 

        self.All_phi_ini=Phi_ini
        self.All_flag_ini=all_ini_flags_matrix
        self.All_interfaces_ini=all_interfaces
        Phi_0=np.asarray(tf.reduce_sum(self.All_interfaces_ini, axis=0))
        self.interface_indices= np.where(np.logical_and(Phi_0.flatten() > 0.9, Phi_0.flatten() <= 1))[0]

        self.phases_ini_indexes=phases_ini_indexes
      
        self.X_phi_test = X_phi_test
        self.X_ini_all_sub_domain=X_ini_train_all
        
        self.phi_ini_all_sub_domain=phi_ini_train_all
       
        self.X_ini_train_all=X_ini_train_all
        self.N_ini =N_ini
        self.X_ini=np.random.choice(len(self.X_ini_train_all), size=int(self.N_ini), replace=True)
        
        self.phi_ini_train_all=phi_ini_train_all
        
        self.indices_ini = np.random.choice(len(self.X_ini_all_sub_domain), size=int(self.N_ini/2), replace=True)
        self.X_ini_all_sub_domain_reduced=X_ini_train_all[self.indices_ini]
        self.phi_ini_all_sub_domain_reduced=phi_ini_train_all[self.indices_ini]

        self.lb = lb
        self.ub = ub
        self.mu = mu
        self.sigma = sigma
        self.delta_g = delta_g

        self.eta = eta
        self.layers = layers
        self.Nx=Nx
        self.Ny=Ny
        self.Nt=Nt
        self.X=X
        self.Y=Y
        X,Y = np.meshgrid(x,y)
        self.X_flat = X.flatten()
        self.Y_flat = Y.flatten()
        
        self.x=x
        self.y=y
        self.dx=(self.ub[0] - self.lb[0]) / (self.Nx - 1)
        self.dy=(self.ub[1] - self.lb[1]) / (self.Ny - 1)
        
        
        self.num_phases=num_phases
        
        # Master PINN limits
        self.abs_x_min=0  # to update and use during the minibatching (these points are the corners of the global space domain)
        self.abs_x_max=0
        self.abs_y_min =0
        self.abs_y_max=0      
        
        self.t_min=self.lb[2]
        self.t_max=self.ub[2]   
        
        self.f=1
        self.ic=1
        self.bc=1   
        self.scipy_max_iter=750
        self.alpha=1 # to incre5ase scipy iterations when reshuffling 
        self.lr=0.000001 #0.000001
        self.precision=tf.float64
        self.precision_="float64"
        self.thresh=0.
        self.increase_pts_if_reshuffle=1.1
        self.thresh_interface=0.1
        self.coef_reduction=16
        
        self.N_batches=np.copy(N_batches) # Total number of batches, default initialization 
        self.min_batch_numbers=min_batch_numbers
        self.Nbr_f_pts_max_per_batch=np.copy(Nbr_f_pts_max_per_batch)# maximum number  Collocation points per batch for Scipy optimizer 
        self.Nbr_f_pts_min_per_batch = np.copy(Nbr_f_pts_min_per_batch)# minimum number of Collocation points per batch for Scipy optimizer 
        self.N_ini_max_per_batch=np.copy(N_ini_max_per_batch) #maximum number of IC points per batch for Scipy optimizer 
        self.N_ini_min_per_batch=np.copy(N_ini_min_per_batch)
        self.Nfeval_master=multiprocessing.Value('i', 1)
        self.Nf=100
        
   
        
        
        
        
        # workers pinns  
        self.pinns=[]
        self.pinn_data_for_scipy = []# minibatches for each pinn for scipy optimizer 
        self.batch_Xf_for_pred= tf.Variable([], dtype=self.precision, trainable=False)
        self.limits=[]
        self.idx_batch=multiprocessing.Value('i', 1000)  # to be accessed in multiprocessing
        self.idx_pinn=multiprocessing.Value('i', 1000)  # to be accessed in multiprocessing
        self.flag=0
        self.list_loss_scipy = []
        self.all_indices_sampled= []
        self.Nfeval = multiprocessing.Value('i', 0) 
        self.order= multiprocessing.Value('i', 0) 
        self.pinns_adam=[]
        self.pinns_adam_above_thresh=[]
        self.pinns_adam_below_thresh=[]
        self.percentage_inter_points=0
        self.old_limits=[]  # store limits of pinns selected at each of the pyramid
        self.old_indexes=[] # store associated indexes
        self.batch_Xf=[]
        self.batch_X_ini=[]
        self.batch_phi_ini=[]
        self.X_ini_all_sub_domain=[]
        self.phi_ini_all_sub_domain=[]
        self.far_int_batch_X_ini=[]
        self.far_int_batch_phi_ini=[]
        self.batch_phi_ini_all=[]
        self.reseve_weights= []  # each worker pinn will get a reserve weights (from the second level of the PYRAMID)
        self.I_beta = tf.constant(0.0, dtype=tf.float64)
        self.I_beta_previous=tf.zeros_like(self.batch_phi_ini)
        # ==> to use to converge
        self.l1_lambda =0.001
        self.l2_lambda =0.001
        self.batch_indices=[] 
        self.all_ini_indices =[] 
        self.flag_one_side=0 
        self.flag_grain=0
        self.flag_no_grain=0
        self.loss_value = multiprocessing.Value('d', 1.0)

       
       
       
        self.W = []  #Weights and biases
        self.parameters = 0 #total number of parameters
        
        for i in range(len(self.layers)-1):
            
            input_dim = layers[i]
            output_dim = layers[i+1]
            #Xavier standard deviation 
            std_dv = np.sqrt((2.0/(input_dim + output_dim)))

            #weights = normal distribution * Xavier standard deviation + 0
            w = tf.random.normal([input_dim, output_dim], dtype = self.precision_) * std_dv
                       
            w = tf.Variable(w, trainable=True, name = 'w' + str(i+1))

            b = tf.Variable(tf.cast(tf.zeros([output_dim]), dtype = self.precision_), trainable = True, name = 'b' + str(i+1))
                    
            self.W.append(w)
            self.W.append(b)
            
            self.parameters +=  input_dim * output_dim + output_dim

        # Define the Adam optimizer
        self.optimizer_Adam = tf.keras.optimizers.Adam(learning_rate=self.lr) 
 
        self.PRE_POST=PrePost(X=X, T=T, lb=lb, ub=ub,Nx=self.Nx,Ny=self.Ny,dx=self.dx,dy=self.dy, x=x, y=y,eta=eta, phi_true=phi_sol)
        """
        # Define the Scipy L-BFGS-B optimizer
        self.scipy_optimizer=scipy.optimize(fun = self.optimizerfunc, 
                                          x0 = init_params, 
                                          args=(), 
                                          method='L-BFGS-B', 
                                          jac= True,        # If jac is True, fun is assumed to return the gradient along with the objective function
                                          callback = PINN.optimizer_callback, 
                                          options = {'disp': None,
                                                    'maxcor': 200, 
                                                    'ftol': 1 * np.finfo(float).eps,  #The iteration stops when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol
                                                    'gtol': 5e-8, 
                                                    'maxfun':  100, 
                                                    'maxiter': 100,
                                                    'iprint': -1,   #print update every 50 iterations
                                                    'maxls': 50})      
        """ 
        # minor check 
        #phi=self.All_phi_ini[3]
        #print(self.All_phi_ini.shape)
        #plt.imshow(phi[0])
        #print(self.x)

        #print(self.X.shape,Y.shape)
        # Flatten phi and convert it to a single column array
        #phi = phi.flatten().reshape(-1, 1)

        #plt.scatter(self.X_flat, self.Y_flat, cmap=plt.get_cmap('viridis'), c=phi)
        #plt.show()

    ###############################################
    def evaluate(self,X):
        lb = tf.cast(self.lb, self.precision)
        ub = tf.cast(self.ub, self.precision)
        lb = tf.reshape(lb, (1, -1))  # Shape: (1, 3)
        ub = tf.reshape(ub, (1, -1))  # Shape: (1, 3)
        X = tf.cast(X, self.precision)

        H = (X - lb)/(ub - lb) 

        for l in range(0,len(self.layers)-2):
            W = self.W[2*l]
            b = self.W[2*l+1]
            H = tf.nn.tanh(tf.add(tf.matmul(H, W), b)) 

        W = self.W[-2]
        b = self.W[-1]
        Y = tf.math.add(tf.matmul(H, W), b) # For regression, no activation to last layer
        #scaled_Y = 0.5 * (Y + 1.0)
        Y = tf.nn.sigmoid(Y) # apply sigmoid activation function
        del lb, ub, H, X, W,b
        return Y #scaled_Y
    ###############################################
    def get_weights(self):
        parameters_1d = []  # [.... W_i,b_i.....  ] 1d array
        
        for i in range (len(self.layers)-1):
            
            w_1d = tf.reshape(self.W[2*i],[-1])   #flatten weights 
            b_1d = tf.reshape(self.W[2*i+1],[-1]) #flatten biases
            
            parameters_1d = tf.concat([parameters_1d, w_1d], 0) #concat weights 
            parameters_1d = tf.concat([parameters_1d, b_1d], 0) #concat biases
        del w_1d,b_1d
        return parameters_1d  
    ############################################### 
    def set_weights(self,parameters):
        for i in range (len(self.layers)-1):

            shape_w = tf.shape(self.W[2*i]).numpy() # shape of the weight tensor
            size_w = tf.size(self.W[2*i]).numpy() #size of the weight tensor 
            
            shape_b = tf.shape(self.W[2*i+1]).numpy() # shape of the bias tensor
            size_b = tf.size(self.W[2*i+1]).numpy() #size of the bias tensor 
                        
            pick_w = parameters[0:size_w] #pick the weights 
            self.W[2*i].assign(tf.reshape(pick_w,shape_w)) # assign  
            parameters = np.delete(parameters,np.arange(size_w),0) #delete 
            
            pick_b = parameters[0:size_b] #pick the biases 
            self.W[2*i+1].assign(tf.reshape(pick_b,shape_b)) # assign 
            parameters = np.delete(parameters,np.arange(size_b),0) 
        del parameters, shape_w,size_w,pick_w
    ############################################### 
    def release_memory(self, quarters_max_indices):
        for pinn in self.pinns:
            if pinn.idx_pinn not in quarters_max_indices:
                del pinn
    ############################################### 
    def Initialize_pinns(self, path_weights_all_pinns):
        self.re_Initialize_pinns(flag_epoch_0=True)
        
        # Iterate over PINNs to create parent and child folders
        for pinn in self.pinns:
            pinn_idx_batch = pinn.idx_batch
            pinn_idx_pinn = pinn.idx_pinn
            
            # folder (pinn.idx_batch)
            parent_folder = os.path.join(path_weights_all_pinns, f"batch_{pinn_idx_batch}")
            if not os.path.exists(parent_folder):
                os.makedirs(parent_folder)
            
            #  folder (pinn.idx_pinn) within the parent folder
            child_folder = os.path.join(parent_folder, f"pinn_{pinn_idx_pinn}")
            if not os.path.exists(child_folder):
                os.makedirs(child_folder)
    ############################################### 
    def re_Initialize_pinns(self, flag_epoch_0=False):
        if flag_epoch_0:
            tf.print("\n ! Initilization of all workers pinns \n")
        else:
            tf.print("\n ! Re-Initilization of all workers pinns \n")
        pinns = []  # List to store the PINN objects
        for batch_idx in range(self.N_batches):
            for pinn_id in range(self.num_phases):
                pinn = Sequentialmodel(layers=self.layers, X_f_train=self.X_f, X_ini_train=self.X_ini,\
                                        phases_ini_indexes=self.phases_ini_indexes,all_ini_flags_matrix=self.All_flag_ini,\
                                        Phi_ini=self.All_phi_ini,phi_ini_train=self.phi_ini, N_ini=self.N_ini,X_phi_test=self.X_phi_test,\
                                        X_ini_train_all=self.X_ini_train_all, phi_ini_train_all=self.phi_ini_train_all,\
                                            all_interfaces=self.All_interfaces_ini,\
                                        X_lb_train=self.X_lb, X_ub_train=self.X_rtb,\
                                        X_ltb_train=self.X_ltb, X_rtb_train=self.X_rtb,\
                                        X=None,Y=None,T=None,x=self.x,y=self.y,lb=self.lb, ub=self.ub, mu=self.mu, sigma=self.sigma, delta_g=self.delta_g,\
                                            eta=self.eta,Nx=self.Nx,Ny=self.Ny,Nt=self.Nt,phi_sol=None,pinns =self.pinns,num_phases=self.num_phases,
                                        N_batches=self.N_batches,
                                        min_batch_numbers = self.min_batch_numbers,\
                                        Nbr_f_pts_max_per_batch=self.Nbr_f_pts_max_per_batch,\
                                        Nbr_f_pts_min_per_batch=self.Nbr_f_pts_min_per_batch,\
                                        N_ini_max_per_batch=self.N_ini_max_per_batch,\
                                        N_ini_min_per_batch=self.N_ini_min_per_batch)   
                                
                #pinn.set_weights(weights_loaded)  # Inherit weights from the master PINN
                pinn.order=batch_idx * self.num_phases + pinn_id
                pinn.set_weights(self.get_weights())
                pinn.idx_batch = batch_idx 
                pinn.idx_pinn = str(batch_idx).zfill(2)  + str(pinn_id).zfill(2) # if two-digit representation
                pinn.idx_phase =str(pinn_id).zfill(2)
                pinns.append(pinn) 
                #tf.print("pinn.idx.batch ", pinn.idx_pinn)

                
        self.pinns=pinns 
        del pinns
    
############################################### 
    def find_pinn_by_idx_pinn(self,desired_idx_pinn, target_pinns=None):
        if target_pinns is None:
            target_pinns= np.copy(self.pinns) 
        corresponding_pinn = None
        for pinn in target_pinns:
            if pinn.idx_pinn == desired_idx_pinn:
                corresponding_pinn = pinn
                break
        del target_pinns
        return corresponding_pinn
    ############################################### 
    def find_pinn_by_idx_batch(self,desired_idx_batch, idx_pinn, target_pinns=None):
        if target_pinns is None:
            target_pinns= np.copy(self.pinns) 
        corresponding_pinns = []
        for pinn in target_pinns:
            if pinn.idx_batch == desired_idx_batch and  pinn.idx_pinn != idx_pinn:
                corresponding_pinns.append(pinn)
        del target_pinns
        return corresponding_pinns
    ###############################################     
    def test_IC(self,N_batches,pathOutput):

        num_x_intervals = int(np.sqrt(N_batches))
        num_y_intervals = int(np.sqrt(N_batches))
        step_x = max(self.x) / num_x_intervals
        step_y = max(self.y) / num_x_intervals
        step_x = max(self.x) / num_x_intervals
        step_y = max(self.y) / num_x_intervals  

        # predcitions 
        for phase_idx in range(self.num_phases):
            n=len(self.X_ini_train_all)
            N=self.num_phases
            X_phi_test= self.X_ini_train_all[int(n*phase_idx/N)+1:int(n*(phase_idx+1)/N)]
            phi_test= self.phi_ini_train_all[int(n*phase_idx/N)+1:int(n*(phase_idx+1)/N)]
            X_phi_test_subsets = []
            
            # Predict for each phase
            fig, ax = plt.subplots()
            for counter, pinn in enumerate(self.pinns):
                batch_idx = counter // self.num_phases
                pinn_id = counter % self.num_phases
                
                t_min, t_max = self.t_min, self.t_max
                i = batch_idx  // num_x_intervals
                j = batch_idx %  num_x_intervals
                x_min = i * step_x
                x_max = (i + 1) * step_x
                y_min = j * step_y
                y_max = (j + 1) * step_y
                
                if  pinn_id==phase_idx:
                    indices = np.where(
                        (X_phi_test[:, 0] >= x_min) &
                        (X_phi_test[:, 0] <= x_max) &
                        (X_phi_test[:, 1] >= y_min) &
                        (X_phi_test[:, 1] <= y_max)
                    )
                    
                    X_phi_test_sub = X_phi_test[indices] 
                    X_phi_test_sub[:,2]=t_max
                    phi_test_sub = phi_test[indices]
                                
                    phi_pred_t_min  = pinn.evaluate(X_phi_test_sub)
                    
                    phi=phi_pred_t_min
                    plt.scatter(X_phi_test_sub[:, 0], X_phi_test_sub[:, 1], cmap=plt.get_cmap('viridis'), c=phi,vmin=0,vmax=1)
                    x_avg = (x_min + x_max) / 2
                    y_avg = (y_min + y_max) / 2
                    plt.text(x_avg, y_avg, f"{pinn.idx_pinn}", color='black', ha='right', va='bottom')
                    
            plt.colorbar()

            fig_name = f"Workers_Pred_Phase_{phase_idx}_at_time_interval_tmin_{t_min:.5f}_tmax_{t_max:.5f}.png"
            plt.savefig(os.path.join(pathOutput ,fig_name)) 
            plt.close()
    ############################################### 
    def h(self,phi):
        return (1/np.pi) * ((4*phi-2) * np.sqrt(phi*(1-phi)) + np.arcsin(2*phi-1)) 
    ###############################################     
    def dh_dphi(self,phi, eta):
        return (8/np.pi) * np.sqrt(phi*(1-phi))
    ###############################################
    def h_term(self,phi):
        try:
            square_root_term=    tf.math.sqrt(phi * (1 - phi))
            #square_root_term=tf.math.sqrt(tf.math.abs(phi) * tf.math.abs(1 - phi))
        except ValueError:
            raise ValueError("Cannot calculate the square root of a negative number")
        else:
            return np.pi/self.eta * square_root_term

    ############################################### 
    def free_energy(self,eta, phi, sigma, h, delta_g): 
        # Calculate the gradient of phi using central differences
        phi_x, phi_y = np.gradient(phi, axis=(0, 1))
        gradient_phi_squared = phi_x**2 + phi_y**2
            
        # Interfacial energy contribution
        f_int = 4 * sigma / eta * (phi * (1 - phi) + (eta**2 / np.pi**2) * gradient_phi_squared)
        
        # Non-interfacial contribution
        f_non_int = - h(phi) * delta_g
        
        # Total free energy
        f = f_int + f_non_int    
        return np.sum(f), np.sum(f_int), np.sum(f_non_int)  
    ############################################### 
    def energy_derivative(self,phi, mu, sigma, eta, delta_g,h): 
        phi_dot = dphi_dt(phi, mu, sigma, eta, delta_g)
        grad_phi = np.gradient(phi, axis=(0, 1))
        grad_phi_dot = np.gradient(phi_dot, axis=(0, 1))
        
        term_1_1 = (4 * sigma / eta) * phi_dot * (1 - 2 * phi)
        #term_1_2 = (4 * sigma / eta) * (eta**2 / np.pi**2) * 2 *  (np.asarray(grad_phi) * np.asarray(grad_phi_dot)).sum()  
        term_1_2 = (4 * sigma / eta) * (eta**2 / np.pi**2) * 2 * np.dot(np.asarray(grad_phi).flatten(), np.asarray(grad_phi_dot).flatten())

        term_2 =dh_dphi(phi,eta) * phi_dot * delta_g
        
        energy_deriv = np.asarray(term_1_1).sum() + term_1_2+ np.asarray(term_2).sum()
        
        return energy_deriv
    ############################################### 
    def phi_term(self,phi, mu, sigma, eta, delta_g): 
        phi_dot = compute_phi_dot(phi, mu, sigma, eta, delta_g, dx)
        grad_phi = np.gradient(phi, axis=(0, 1))
        grad_phi_dot = np.gradient(phi_dot, axis=(0, 1))
        
        term_1_1 = (4 * sigma / eta) * phi_dot * (1 - 2 * phi)
        term_1_2 = (4 * sigma / eta) * (eta**2 / np.pi**2) * 2 *  (np.asarray(grad_phi) * np.asarray(grad_phi_dot))
        #term_1_2 =  (4 * sigma / eta) * (eta**2 / np.pi**2) * 2 * np.dot(np.asarray(grad_phi), np.asarray(grad_phi_dot)).sum()

        term_2 = -dh_dphi(phi,eta) * phi_dot * delta_g
        
        phi_term_ = np.asarray(term_1_1).sum() + term_1_2.sum() + np.asarray(term_2).sum()
        return phi_term_
    ###############################################   
    def loss_Energy(self,eta, phi, sigma, h, delta_g, mu, dx, dy):
        dF_dt=0
        energy_deriv = self.energy_derivative(phi, mu, sigma, eta, delta_g,h)
        
        # Compute the residual as the absolute difference between the right-hand side and the global energy derivative
        
        residual = np.abs(dF_dt - energy_deriv)
        #print("here: ", dF_dt ,energy_deriv)
        energy,f_int,f_non_int=self.free_energy(eta, phi,sigma, h, delta_g)
        
        phi_term_=self.phi_term(phi, mu, sigma, eta, delta_g)
        energy_residual = np.abs(np.abs(derivative) - np.abs(phi_term_))
        return energy_residual
    ###############################################
    def loss_IC(self,x_ini,phi_ini): 

        phi_ini_pred=self.evaluate(x_ini)  
        MSE_loss_IC = tf.reduce_mean(tf.square(phi_ini-phi_ini_pred))
        
        exclude_batch = tf.reduce_any(tf.logical_and(phi_ini > 0, phi_ini < 1))
        if exclude_batch:
            alpha = 1.0 
        else:
            alpha =1 

        loss_IC = alpha * MSE_loss_IC 
        global epoch 
        #tf.print(epoch, self.idx_pinn, MSE_loss_IC)
        del phi_ini_pred, x_ini,phi_ini
        return loss_IC
    ###############################################
    def loss_BC(self,X_lb,X_ub,X_ltb,X_rtb,abs_x_min,abs_x_max,abs_y_min,abs_y_max):
        #tf.print("abs_x_min,abs_x_max,abs_y_min,abs_y_max: ", abs_x_min,abs_x_max,abs_y_min,abs_y_max)   
        #tf.print("X_ltb: ", X_ltb)
        #tf.print("X_ltb[:,1]: ", X_ltb[:,1])

        #X_ltb = tf.cast(X_ltb, dtype=self.precision)
        #X_rtb = tf.cast(X_rtb, dtype=self.precision)
        #X_ub = tf.cast(X_ub, dtype=self.precision)
        #X_lb = tf.cast(X_lb, dtype=self.precision)
  
        if flag_scipy==True:
            return 0
  
        if (len(X_lb)==0) or (len(X_ub)==0) or (len(X_ltb)==0) or (len(X_rtb)==0) :
            tf.print(" !!! !!! !!! !!! :\n !!! increase your BC points !!!:\n !!! !!! !!! !!! :\n") 

        flag_lb=0
        flag_ub=0
        flag_ltb=0
        flag_rtb=0

        if tf.reduce_min(X_lb[:, 0])==abs_x_min:
            flag_ltb=1
            pred_ltb=self.evaluate(X_ltb)
            loss_ltb=tf.reduce_mean(tf.square(pred_ltb))
            del pred_ltb
        else:
            loss_ltb= 0
            flag_ltb=0

        if tf.reduce_max(X_lb[:, 0])==abs_x_max:
            flag_rtb=1
            pred_rtb=self.evaluate(X_rtb)
            loss_rtb=tf.reduce_mean(tf.square(pred_rtb))
            del pred_rtb
        else:
            loss_rtb= 0
            flag_rtb=0
        
        if tf.reduce_min(X_ltb[:, 1])==abs_y_min:
            flag_lb=1
            pred_lb=self.evaluate(X_lb)
            loss_lb=tf.reduce_mean(tf.square(pred_lb))
            del pred_lb
        else:
            loss_lb= 0
            flag_lb=0 

        if tf.reduce_max(X_ltb[:, 1])==abs_y_max:
            flag_ub=1
            pred_ub=self.evaluate(X_ub)
            loss_ub=tf.reduce_mean(tf.square(pred_ub))
            del pred_ub

        else:
            loss_ub= 0
            flag_ub=0
                
        """
        * ***** ub *****      
        *              *
        *              *
        *              *
       ltb     X_f    rtb
        *              *    Y
        *              *    ^
        *              *    | 
        * ***** lb *****     –– > X  
        """        
        #just activate with 1 epoch training for initial check
        """
        if  flag_lb==1 and flag_ltb==1:    
            plt.scatter(X_lb[:, 0], X_lb[:, 1], color="r")
            plt.scatter(X_ltb[:, 0], X_ltb[:, 1], color="b")
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

        if  flag_ub==1 and flag_ltb==1:    
            plt.scatter(X_ub[:, 0], X_ub[:, 1], color="g")
            plt.scatter(X_ltb[:, 0], X_ltb[:, 1], color="b")
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

        if  flag_lb==1 and flag_rtb==1:     
            plt.scatter(X_lb[:, 0], X_lb[:, 1], color="m")
            plt.scatter(X_rtb[:, 0], X_rtb[:, 1], color="b")
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

        if  flag_ub==1 and flag_rtb==1:    
            plt.scatter(X_ub[:, 0], X_ub[:, 1], color="c")
            plt.scatter(X_rtb[:, 0], X_rtb[:, 1], color="m")
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()
        """    
        loss_BC =  loss_lb+ loss_ub+ loss_ltb+loss_rtb
        #tf.print("loss_BC: ", loss_BC)
        del X_lb,X_ub,X_ltb,X_rtb
        return loss_BC
    ###############################################
    ###############################################
    def get_column_row(self,batch_index, N_batches):
        grid_size = int(np.sqrt(N_batches))
        Grid = np.arange(N_batches).reshape(grid_size, grid_size, order='F').T
        for i in range(grid_size):
            for j in range(grid_size):
                index_batch = i * grid_size + j
                #print("i,j: ",i,j)
                if index_batch == batch_index:
                    row_indices=Grid[:,j]  # Get row indices for the batch
                    column_indices=Grid[i,:] # Get column indices for the batch
                    pos_col, pos_row= i,j
        #tf.print("row_indices for batch", batch_index, ": ", row_indices)
        #tf.print("column_indices for batch", batch_index, ": ", column_indices)
        return column_indices, row_indices,  pos_col, pos_row
    ###############################################
    def get_neighboring_indices(self,pinn, N_batches):
        batch_index=int(pinn.idx_batch) 
        pinn_idx=int(pinn.idx_batch)    
        upper_neighbour=-1
        est_neighbour=-1
        grid_size = int(np.sqrt(N_batches))
        col = batch_index % grid_size
        upper_neighbour = batch_index + 1  
        est_neighbour = batch_index + grid_size  #
        
        column, row, pos_col, pos_row= self.get_column_row(batch_index,N_batches)

        if int(upper_neighbour) not in column: 
            upper_neighbour=str(-1)
        else:
            upper_neighbour = str(upper_neighbour).zfill(2) + pinn.idx_pinn[-2:]
            
        if int(est_neighbour) not in row: 
            est_neighbour=str(-1)
        else:
            est_neighbour = str(est_neighbour).zfill(2) + pinn.idx_pinn[-2:]
            
        #tf.print("pinn_index:" ,pinn.idx_pinn, "==> upper, est : ",upper_neighbour,", ", est_neighbour, "pos_col, pos_row: ",pos_col, pos_row)

        return upper_neighbour, est_neighbour, pos_col, pos_row
    ###############################################
    def get_neighboring_indices_west_inner(self, pinn, N_batches):
        batch_index = int(pinn.idx_batch)
        grid_size = int(np.sqrt(N_batches))
        col = batch_index % grid_size
    
        inner_neighbour = batch_index - 1  
        west_neighbour = batch_index - grid_size  
    
        west_neighbour = str(west_neighbour).zfill(2) + pinn.idx_pinn[-2:]
        inner_neighbour = str(inner_neighbour).zfill(2) + pinn.idx_pinn[-2:]
    
        return west_neighbour, inner_neighbour,  grid_size-1, grid_size-1
    ###############################################
    def get_X_ini_all(self,pinn,Master_PINN,pos_col, pos_row,N_batches,X,Y, case): 
        thresh=0.5
        thresh_interface=pinn.thresh_interface
        self_X_ini_all_sub_domain= Master_PINN.X_ini_all_sub_domain[pinn.all_ini_indices]
        self_phi_ini_all_sub_domain= Master_PINN.phi_ini_all_sub_domain[pinn.all_ini_indices]
        interfacial_indices = np.where(np.logical_and(self_phi_ini_all_sub_domain >= thresh_interface, self_phi_ini_all_sub_domain <= 1-thresh_interface))[0]
        self_X_ini_all_sub_domain=self_X_ini_all_sub_domain[interfacial_indices]
        self_phi_ini_all_sub_domain=self_phi_ini_all_sub_domain[interfacial_indices]
        
        if case =="condition_upper":
            condition=abs(self_X_ini_all_sub_domain[:, 1] - (pos_row+1)* Y / np.sqrt(N_batches))<  thresh
                
        if case =="condition_east":
            condition=abs(self_X_ini_all_sub_domain[:, 0] -  (pos_col+1)*X / np.sqrt(N_batches))<  thresh
            """ 
            #Debug
            if pinn.idx_pinn=="0000":
                plt.figure()
                scatter=plt.scatter(self_X_ini_all_sub_domain[:, 0], self_X_ini_all_sub_domain[:, 1], c=self_phi_ini_all_sub_domain, cmap='viridis', marker='o', label='0_1')
                cbar = plt.colorbar(scatter, shrink=0.35)
                cbar.set_label('phi')
                plt.title(f"pinn_{self.idx_pinn}")
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.savefig("subset_east")
                plt.close()
                tf.print(stop_here)
            """
        if case =="condition_inner":
            condition=abs(self_X_ini_all_sub_domain[:, 1] -  (pos_row)* Y / np.sqrt(N_batches))<  thresh
        if case =="condition_west":
            condition=abs(self_X_ini_all_sub_domain[:, 0] -   (pos_col)*X / np.sqrt(N_batches))<  thresh
        
        selected_indices=np.where(condition)[0]
        subset=self_X_ini_all_sub_domain[selected_indices]
        # take exact boundary
        if case =="condition_upper":
            subset[:, 1] = (pos_row+1)* Y / np.sqrt(N_batches)
        if case =="condition_east":
            subset[:, 0] = (pos_col+1)*X / np.sqrt(N_batches)
        if case =="condition_inner":
            subset[:, 1] =  (pos_row)* Y / np.sqrt(N_batches)
        if case =="condition_west":
            subset[:, 0] =   (pos_col)*X / np.sqrt(N_batches)
            
        # update time
        subset=np.unique(subset, axis=0)
        updated_time_column = np.random.uniform(pinn.t_min, pinn.t_max, subset.shape[0])
        subset[:, 2]= updated_time_column        
        return np.unique(subset, axis=0), selected_indices
    ###############################################
    def loss_BC_custom(self,X_lb,X_ub,X_ltb,X_rtb,abs_x_min,abs_x_max,abs_y_min,abs_y_max):
        
        with tf.device('/CPU:0'):
            lock= multiprocessing.Lock()
            lock.acquire()        
            global epoch   
            global Master_PINN   
            pinns=Master_PINN.pinns      
            
            global bool_flag_continuity
            flag_continuity= bool_flag_continuity 
            
            global flag_scipy
            
            if flag_scipy==True:
              return 0
            

            # get pinns_beta (pinns in the same bath but with different phases
            pinns_beta=self.find_pinn_by_idx_batch(self.idx_batch, self.idx_pinn,target_pinns=Master_PINN.pinns)
            
            # get X_f_common : common batch for pinns handling different phases in the same bacth
            if int(self.idx_phase) == 0:
                X_f_common=self.batch_Xf
            else: 
                for pinn in pinns_beta:
                    if int(pinn.idx_phase)==0:
                        X_f_common=pinn.batch_Xf
                        break          

            if  len(self.pinn_data_for_scipy) ==0  : 
                tf.print("! Caution --- no data for scipy for the batch ",self.idx_pinn, " at Epoch ",int(epoch))
                return 0

            X = self.x.max()
            Y = self.y.max()
            t_min = self.t_min
            t_max = self.t_max
            #loss_BC = 0 # to return if no neighbourours 
            all_losses = []
            N_batches=len(Master_PINN.pinns)//self.num_phases
            # Loop through each batch index
            for pinn in Master_PINN.pinns:
                if self.idx_phase == pinn.idx_phase and self.idx_pinn != pinn.idx_pinn and self.idx_batch != pinn.idx_batch: 
                    loss_BC = 0
                    upper_neighbor, east_neighbor,pos_col, pos_row = self.get_neighboring_indices(self,N_batches)  # You need to implement this function
                    if str(upper_neighbor)!= "-1":
                        #tf.print("batch_index:" ,self.idx_pinn, "==> upper, est : ",upper_neighbor,", ", east_neighbor, "pos_col, pos_row: ",pos_col, pos_row)
                        pinn_upper=self.find_pinn_by_idx_pinn(upper_neighbor, Master_PINN.pinns)
                        subset_upper, indices_upper= self.get_X_ini_all(self,Master_PINN,pos_col, pos_row,N_batches,X,Y,"condition_upper")

                        if len(indices_upper)==0:
                            tf.print("  !!! no  BC points (upper bound) for the pinn ",self.idx_pinn)
                            return 0
                        self_pred=self.evaluate(subset_upper)
                        upper_pred=pinn_upper.evaluate(subset_upper)
                        loss_BC_upper=tf.reduce_mean(tf.square(self_pred-upper_pred))
                        loss_BC+=loss_BC_upper
    
                        # DEBUG
                        """ 
                        tf.print(subset_upper)
                        plt.figure()
                        plt.scatter(X_f_common[:, 0], X_f_common[:, 1], c=X_f_common[:, 2], cmap='magma', marker='o')
                        scatter=plt.scatter(subset_upper[:, 0], subset_upper[:, 1], c=self_pred, cmap='viridis', marker='o')
                        cbar = plt.colorbar(scatter, shrink=0.35)
                        cbar.set_label('phi')
                        plt.title(f"pinn_{self.idx_pinn}")
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        plt.savefig(f"subset_upper_pinn_{self.idx_pinn}")
                        plt.close()
                        tf.print(subset_upper)
                        tf.print(stop_here)
                        """ 

                    if str(east_neighbor)!= "-1":
                        subset_est=np.copy(X_f_common)
                        pinn_east=self.find_pinn_by_idx_pinn(east_neighbor, target_pinns=Master_PINN.pinns)
                        subset_east, indices_east= self.get_X_ini_all(self,Master_PINN,pos_col, pos_row,N_batches,X,Y,"condition_east")
                        
                        if len(indices_east)==0:
                            tf.print(" !!! no  BC points (east bound) for the pinn ",self.idx_pinn)
                            return 0  

                        self_pred=self.evaluate(subset_east)
                        east_pred=pinn_east.evaluate(subset_east)
                        loss_BC_est=tf.reduce_mean(tf.square(self_pred-east_pred))
                        loss_BC+=loss_BC_est
                        # DEBUG
                        """ 
                        plt.figure()
                        plt.scatter(X_f_common[:, 0], X_f_common[:, 1], c=X_f_common[:, 2], cmap='magma', marker='o', label='batch ')
                        scatter=plt.scatter(subset_east[:, 0], subset_east[:, 1], c=self_pred, cmap='viridis', marker='o', label='0_1')
                        cbar = plt.colorbar(scatter, shrink=0.35)
                        cbar.set_label('phi')
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        plt.savefig(f"subset_east_pinn_{self.idx_pinn}")
                        plt.close()  
                        tf.print(subset_east)
                        """ 
                        
                    if str(east_neighbor)== "-1" and str(upper_neighbor)== "-1":  # batch in the upper right corner
                        west_neighbour, inner_neighbour, pos_col, pos_row =self.get_neighboring_indices_west_inner(self, N_batches)
                        
                        # subset_inner (inner neighbour)
                        pinn_inner=self.find_pinn_by_idx_pinn(inner_neighbour, pinns)
                        subset_inner, indices_inner= self.get_X_ini_all(self,Master_PINN,pos_col, pos_row,N_batches,X,Y,"condition_inner")
                        
                        if len(indices_inner)==0:
                            tf.print(" !!! no  BC points (inner bound) for the pinn ",self.idx_pinn)
                            return 0  
    
                        self_pred=self.evaluate(subset_inner)
                        inner_pred=pinn_inner.evaluate(subset_inner)
                        loss_BC_inner=tf.reduce_mean(tf.square(self_pred-inner_pred))
                        loss_BC+=loss_BC_inner
                        # DEBUG  
                        """                      
                        plt.figure()
                        plt.scatter(X_f_common[:, 0], X_f_common[:, 1], c=X_f_common[:, 2], cmap='magma', marker='o', label='batch ')
                        scatter=plt.scatter(subset_inner[:, 0], subset_inner[:, 1], c=self_pred, cmap='viridis', marker='o', label='0_1')
                        plt.title(f"pinn_{self.idx_pinn} - inner_neighbour{pinn_inner.idx_pinn}")
                        cbar = plt.colorbar(scatter, shrink=0.35)
                        cbar.set_label('phi')
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        plt.savefig(f"subset_inner_pinn_{self.idx_pinn}")
                        plt.close() 
                        tf.print("subset_inner" ,subset_inner)
                        """
                        
                        # subset_west (west neighbour)
                        pinn_west=self.find_pinn_by_idx_pinn(west_neighbour, pinns)
                        subset_west, indices_west= self.get_X_ini_all(self,Master_PINN,pos_col, pos_row,N_batches,X,Y,"condition_west")    
                        
                        if len(indices_west)==0:
                            tf.print(" !!! no  BC points (west bound) for the pinn ",self.idx_pinn)
                            return 0
        
                        self_pred=self.evaluate(subset_west)
                        west_pred=pinn_west.evaluate(subset_west)
                        loss_BC_west=tf.reduce_mean(tf.square(self_pred-west_pred))
                        loss_BC+=loss_BC_west
                        # DEBUG
                        """
                        plt.figure()
                        plt.scatter(X_f_common[:, 0], X_f_common[:, 1], c=X_f_common[:, 2], cmap='magma', marker='o', label='batch ')
                        scatter=plt.scatter(subset_west[:, 0], subset_west[:, 1], c=self_pred, cmap='viridis', marker='o', label='0_1')
                        plt.title(f"pinn_{self.idx_pinn}")
                        cbar = plt.colorbar(scatter, shrink=0.35)
                        cbar.set_label('phi')
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        plt.savefig(f"subset_west_pinn_{self.idx_pinn}")
                        tf.print("subset_west" ,subset_west)
                        plt.close()  
                        """
                        
            if tf.reduce_any(tf.math.is_nan(tf.cast(loss_BC, dtype=tf.float32))):
                tf.print(" !!!!!! Check BC points returning nan losses !!!!!! ", " ==> pinn: ", self.idx_pinn)
                return 0

            lock.release()
            return loss_BC
    ###############################################
    def Sigma(self,phi_alpha,phi_phi_beta):
        return self.sigma # we assume the same interfacial energies for all phases
    ###############################################       
    def I_phi(self,lap_phi_alpha,phi_alpha,Prefactor):
        return (lap_phi_alpha +Prefactor*phi_alpha )
    ###############################################
    #@tf.function
    def loss_PDE(self, X_f, phi_ini):
        lock= multiprocessing.Lock()
        lock.acquire()

        X_f=self.batch_Xf

        g = tf.Variable(X_f, dtype=self.precision, trainable=False) 
        x, y, t= tf.split(g, num_or_size_splits=3, axis=1)
        Prefactor = np.pi**2 / self.eta**2
        thresh_interface=self.thresh_interface

        phase_fields= []
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(t)
            g = tf.concat([x, y, t], axis=1)
            
            phi_alpha= self.evaluate(g)   
            phi_alpha = tf.clip_by_value(phi_alpha, 0.0, 1.0)

            loss_f = 0.0
            phase_losses = []      
            dPsi_dt=tf.zeros_like(phi_alpha )
            tape.watch(phi_alpha)
            phi_t_alpha = tape.gradient(phi_alpha, t)
            
            phi_x_alpha = tape.gradient(phi_alpha, x)   
            tape.watch(phi_x_alpha)              
            phi_y_alpha = tape.gradient(phi_alpha, y)
            tape.watch(phi_y_alpha)              
            phi_xx_alpha = tape.gradient(phi_x_alpha, x)                   
            phi_yy_alpha = tape.gradient(phi_y_alpha, y)
            lap_phi_alpha = phi_xx_alpha + phi_yy_alpha
  
             
        phi_term = (np.pi**2 / (2 * self.eta**2)) * (2 * phi_alpha - 1)
        right_side_eqn = self.mu * ( self.sigma * ( lap_phi_alpha + phi_term)+ self.h_term(phi_alpha) * self.delta_g )        
        phi_t=phi_t_alpha
        f = phi_t - right_side_eqn
        # PDE loss
        loss_f += tf.reduce_mean(tf.square(f))

        loss_total = loss_f       

        lock.release()
        return loss_total      
    ###############################################   
    def get_denoising_loss(self) :   
        loss_far_int =0
        
        if self.flag_one_side == 1:  #  case if interface with only one side ( grain or no-grain)
            phi_pred=self.evaluate(self.far_int_batch_X_ini)
            # no grain
            if self.flag_grain == 0:
                loss_far_int = tf.reduce_mean(tf.square(phi_pred))  # the mean should be close to zero      
            # grain
            else: 
                loss_far_int = tf.reduce_mean(tf.square(1 - phi_pred))  # the mean should be close to one
        
        else:  # three sides : left side ( no grain) -- interface  -- right side (grain)
            condition_left = self.far_int_batch_phi_ini <= self.thresh_interface  # far within the zero phi values
            batch_far_left = self.far_int_batch_X_ini[np.where(condition_left)[0]] # far within the grain
            condition_right = self.far_int_batch_phi_ini >=1- self.thresh_interface
            batch_far_right = self.far_int_batch_X_ini[np.where(condition_right)[0]]
            right_pred=self.evaluate(batch_far_right)
            left_pred=self.evaluate(batch_far_left)
            far_product = right_pred * left_pred
            #far_sum = tf.reduce_sum(tf.concat([right_pred, left_pred], axis=1), axis=1) - 1
            loss_far_int =  tf.reduce_mean(tf.square(left_pred)) +tf.reduce_mean(tf.square(1 - right_pred))

        #tf.print(loss_far_int)
        return loss_far_int
    ###############################################        
    def loss(self,xf,x_ini,x_lb,x_ub,x_ltb,x_rtb,phi_ini,abs_x_min,abs_x_max,abs_y_min,abs_y_max):
        global denoising_loss_  
        denoising_loss= self.get_denoising_loss() if denoising_loss_==True else  0
        loss_IC = self.loss_IC(x_ini,phi_ini)      
        loss_f =  self.loss_PDE(xf,phi_ini)        
        loss_BC =  self.loss_BC_custom(x_lb,x_ub,x_ltb,x_rtb,abs_x_min,abs_x_max,abs_y_min,abs_y_max)        
        loss =  self.f*loss_f +self.ic*loss_IC+self.bc*loss_BC +denoising_loss # 
        #l1_regularization = tf.reduce_sum([tf.reduce_sum(tf.abs(w)) for w in self.trainable_variables])
    

        #l2_regularization = tf.reduce_sum([tf.reduce_sum(tf.square(w)) for w in self.trainable_variables])
    
        # Add the regularization terms to the loss
        total_loss = loss   #+ self.l1_lambda * l1_regularization + self.l2_lambda * l2_regularization   
        del xf,x_ini,x_lb,x_ub,x_ltb,x_rtb,phi_ini
        return loss, loss_BC,loss_IC, loss_f 
    ###############################################
    def optimizerfunc(self,parameters):  # Global Optiöization (Master pinn)
        with tf.device('/CPU:0'):
            global list_loss_scipy
            #global Nfeval
            self.set_weights(parameters)
                
            X_ini =self.X_ini_all_sub_domain[self.indices_ini]
            phi_ini=self.phi_ini_all_sub_domain[self.indices_ini]
            
            X_f = jnp.asarray(self.X_f_sub_domain_scipy, dtype=jnp.float64)
            X_lb = jnp.asarray(self.X_lb_sub_domain, dtype=jnp.float64)
            X_ub = jnp.asarray(self.X_ub_sub_domain, dtype=jnp.float64)
            X_ltb = jnp.asarray(self.X_ltb_sub_domain, dtype=jnp.float64)
            X_rtb = jnp.asarray(self.X_rtb_sub_domain, dtype=jnp.float64)
            X_ini = jnp.asarray(X_ini, dtype=jnp.float64)
            phi_ini = jnp.asarray(phi_ini, dtype=jnp.float64) 

            with tf.GradientTape() as tape:
                tape.watch(self.trainable_variables)
                
                loss_val, loss_BC,loss_IC, loss_f = self.loss(X_f,X_ini,X_lb,X_ub,X_ltb,X_rtb,phi_ini,self.abs_x_min,self.abs_x_max,self.abs_y_min,self.abs_y_max)   
                list_loss_scipy.append([loss_val, loss_BC,loss_IC, loss_f ])
                grads = tape.gradient(loss_val,self.trainable_variables)
                    
            del X_ini,phi_ini,X_f, X_lb, X_ub, X_ltb, X_rtb
            
            grads_1d = [ ] #flatten grads 
            for i in range (len(self.layers)-1):

                grads_w_1d = tf.reshape(grads[2*i],[-1]) #flatten weights 
                grads_b_1d = tf.reshape(grads[2*i+1],[-1]) #flatten biases

                grads_1d = tf.concat([grads_1d, grads_w_1d], 0) #concat grad_weights 
                grads_1d = tf.concat([grads_1d, grads_b_1d], 0) #concat grad_biases

            del grads, grads_w_1d,grads_b_1d
            return loss_val.numpy(), grads_1d.numpy()
    ###############################################
    def optimizer_callback_master(self, parameters):
        with tf.device('/CPU:0'):
            global list_loss_scipy

            # Extract numeric values from TensorFlow tensors
            total_loss = float(list_loss_scipy[-1][0])
            loss_BC = float(list_loss_scipy[-1][1])
            loss_IC = float(list_loss_scipy[-1][2])
            loss_f = float(list_loss_scipy[-1][3])
            
            if self.Nfeval_master.value % 100 == 0:
                formatted_str = 'Iter: {:}, total_loss: {:.3e}, loss_BC: {:.3e}, loss_IC: {:.3e}, loss_f: {:.3e}'.format(self.Nfeval_master.value, total_loss, loss_BC, loss_IC, loss_f)
                tf.print(formatted_str)

            self.Nfeval_master.value += 1 
            return list_loss_scipy
    ###############################################
    #@tf.function()
    def optimize_single_pinn(self, pinn, init_params):
        try:
            # Perform the optimization for a single PINN
            import functools
            func_to_minimize = functools.partial(pinn.optimizerfunc_pinn)
    
            result = scipy.optimize.minimize(fun=func_to_minimize,
                                    x0=init_params,
                                    args=(),
                                    method='L-BFGS-B',
                                    jac=True,
                                    callback=pinn.optimizer_callback,
                                    options={'disp': None,
                                                'maxcor': 50, 
                                                'ftol': 1 * np.finfo(float).eps,
                                                'gtol': 5e-14,
                                                'maxfun':  self.alpha*pinn.scipy_max_iter,
                                                'maxiter':  self.alpha*pinn.scipy_max_iter,
                                                'iprint': -1,
                                                'maxls': 50
                                                })
                                                
            del func_to_minimize  
            return result
        except Exception as e:
            tf.print(f"An exception occurred for pinn {pinn.idx_pinn}")
            tf.print(traceback.format_exc())    
                                                                                            
        return None



    ###############################################
    #@tf.function()
    def optimizerfunc_pinn(self,parameters):
        with tf.device('/CPU:0'):
        
            global pinns
  
            self.set_weights(parameters)
            
            X_f, X_ini_all, X_lb, X_ub, X_ltb, X_rtb, phi_ini_all = self.pinn_data_for_scipy

            num_samples = int(len(X_ini_all) ) # coef if needed 
            if num_samples==0:
                tf.print(" !!! Increase IC points for pinn {0}, num_samples = {1}, len(X_ini_all) = {2} !!!".format(self.idx_pinn, num_samples,len(X_ini_all)))

            #indices_ini = np.random.choice(len(X_ini_all), size=num_samples, replace=False)
            X_ini= self.batch_X_ini #self.X_ini_all_sub_domain # #
            phi_ini=self.batch_phi_ini # self.phi_ini_all_sub_domain  # #
        
            X_f = tf.convert_to_tensor(X_f, dtype=self.precision)
            X_lb = tf.convert_to_tensor(X_lb, dtype=self.precision)
            X_ub = tf.convert_to_tensor(X_ub, dtype=self.precision)
            X_ltb = tf.convert_to_tensor(X_ltb, dtype=self.precision)
            X_rtb = tf.convert_to_tensor(X_rtb,dtype=self.precision)
            X_ini_all = tf.convert_to_tensor(X_ini_all, dtype=self.precision)
            phi_ini_all = tf.convert_to_tensor(phi_ini_all, dtype=self.precision)
            X_ini= tf.convert_to_tensor(X_ini, dtype=self.precision)
            phi_ini= tf.convert_to_tensor(phi_ini, dtype=self.precision)

            with tf.GradientTape() as tape:
                tape.watch(self.trainable_variables)
                
                loss_val, loss_BC,loss_IC, loss_f = self.loss(X_f,X_ini,X_lb,X_ub,X_ltb,X_rtb,phi_ini,self.abs_x_min,self.abs_x_max,self.abs_y_min,self.abs_y_max)   
                self.list_loss_scipy.append([loss_val, loss_BC,loss_IC, loss_f ])
                grads = tape.gradient(loss_val,self.trainable_variables)
                    
            #del X_ini,phi_ini,X_f, X_lb, X_ub, X_ltb, X_rtb

            grads_1d = [ ] #flatten grads 
            for i in range (len(self.layers)-1):
                grads_w_1d = tf.reshape(grads[2*i],[-1]) #flatten weights 
                grads_b_1d = tf.reshape(grads[2*i+1],[-1]) #flatten biases

                grads_1d = tf.concat([grads_1d, grads_w_1d], 0) #concat grad_weights 
                grads_1d = tf.concat([grads_1d, grads_b_1d], 0) #concat grad_biases
    
            #Global_pinns[self.order].set_weights(grads_1d.numpy())
            self.set_weights(grads_1d.numpy())
            
            del grads, grads_w_1d,grads_b_1d
            
            self.list_loss_scipy= self.optimizer_callback(parameters) # each pinn ==> list_loss_scipy 
                    
            return loss_val.numpy(), grads_1d.numpy(), self.list_loss_scipy    
    ###############################################
    #@tf.function()
    def optimizer_callback(self, parameters):  
             
        with self.Nfeval.get_lock():
            self.Nfeval.value += 1
            
            if self.Nfeval.value % 100 == 0:  # Print during scipy iterations 
                print_lock = multiprocessing.Lock()
                print_lock.acquire()
                tf.print("pinn: {:}, Iter: {:d}, total_loss: {:.3e}, loss_BC: {:.3e}, loss_IC: {:.3e},loss_f: {:.3e}".format(self.idx_pinn, self.Nfeval.value, \
                    self.list_loss_scipy[-1][0], self.list_loss_scipy[-1][1], self.list_loss_scipy[-1][2], self.list_loss_scipy[-1][3]))
                print_lock.release()

        return self.list_loss_scipy
    ##############################################
    #@tf.function
    def process_batch(self, batch_X_f, batch_X_ini, batch_X_lb, batch_X_ub,batch_X_ltb,batch_X_rtb, batch_phi_ini,model): #,
        with tf.device('/CPU:0'): 
            with tf.GradientTape() as tape:
                loss, loss_BC,loss_IC, loss_f = model.loss(batch_X_f,batch_X_ini,\
                                                                    batch_X_lb,batch_X_ub,batch_X_ltb,batch_X_rtb,\
                                                                        batch_phi_ini,self.abs_x_min,self.abs_x_max,self.abs_y_min,self.abs_y_max)             
                gradients = tape.gradient(loss, model.trainable_variables)
                model.optimizer_Adam.apply_gradients(zip(gradients, model.trainable_variables))
                
            process_name = multiprocessing.current_process().name
            del tape, gradients
            #tf.print("Processor:", process_name)
            return model.idx_pinn,loss, loss_BC, loss_IC, loss_f
    ###############################################
    def sum_square(self,n):
        # Compute the sum of squares
        result = sum([i**2 for i in range(n)])
        return result
    ###############################################
    def print_final_loss(self,x):
        # Wrap the original callback function to print the final loss value
        def wrapped_callback(x):
            if not wrapped_callback.done:
                self.optimizer_callback(x)
                wrapped_callback.done = True

        wrapped_callback.done = False
        results = self.optimizer_callback(x)

        # Print the final loss value
        tf.print(results.fun)

        return results
    ###############################################
    def process_repository_files_discret_workers_Master(self,epoch,N_batches,\
        path,pathOutput,path_weights_all_pinns,title,filename,flag_Master=False,flag_t_max=True):
    
        num_x_intervals = int(np.sqrt(N_batches))
        num_y_intervals = int(np.sqrt(N_batches))
        step_x = max(self.x) / num_x_intervals
        step_y = max(self.y) / num_x_intervals  
        t_min=self.t_min
        t_max=self.t_max

        for phase_idx in range(self.num_phases):            
            
            # Predict evolution of each phase
            fig, ax = plt.subplots()
                
            for counter, pinn in enumerate(self.pinns):
                
                batch_idx = counter // self.num_phases
                pinn_id = counter % self.num_phases
                n=len(self.X_ini_all_sub_domain)
                N=self.num_phases
                X_ini_all_sub_domain_phase= self.X_ini_all_sub_domain[int(n*pinn_id/N)+1:int(n*(pinn_id+1)/N)] 
            
                if flag_Master==True:
                    pinn=self # to use the same plot function for pinns and Master to save results 
                    
                
                i = batch_idx  // num_x_intervals
                j = batch_idx %  num_y_intervals
                x_min = i * step_x
                x_max = (i + 1) * step_x
                y_min = j * step_y
                y_max = (j + 1) * step_y
                #batch_indices = np.where((X_ini_all_sub_domain_phase[:, 0] >= x_min) & (X_ini_all_sub_domain_phase[:, 0] <= x_max) &
                #                            (X_ini_all_sub_domain_phase[:, 1] >= y_min) & (X_ini_all_sub_domain_phase[:, 1] <= y_max))[0]
                batch_indices=pinn.batch_indices
                X_ini_all_sub_domain_pinn=X_ini_all_sub_domain_phase[batch_indices]
                X_ini_all_sub_domain_pinn[:,2]=self.t_max
                phi_ini_all_sub_domain_pinn=pinn.evaluate(X_ini_all_sub_domain_pinn)
                
                if  int(pinn.idx_phase)==phase_idx:
                    scatter=ax.scatter(X_ini_all_sub_domain_pinn[:, 0], X_ini_all_sub_domain_pinn[:, 1], cmap=plt.get_cmap('viridis'), c=phi_ini_all_sub_domain_pinn,vmin=0,vmax=1)
                    x_avg = (x_min + x_max) / 2
                    y_avg = (y_min + y_max) / 2
                    ax.text(x_avg, y_avg, f"{pinn.idx_pinn}", color='black', ha='right', va='bottom')
                
            plt.colorbar(scatter)
            #plt.tight_layout()
            if flag_Master==False:
                if flag_t_max==False: 
                    fig_name = f"Workers_Pred_Phase_{phase_idx}_at_Epoch_{epoch}_time_interval_tmin_{t_min:.5f}_tmax_{t_max:.5f}_{N_batches}_batches_t_min.png"
                else: 
                    fig_name = f"Workers_Pred_Phase_{phase_idx}_at_Epoch_{epoch}_time_interval_tmin_{t_min:.5f}_tmax_{t_max:.5f}_{N_batches}_batches_t_max.png"
            else:
                if flag_t_max==False: #here we add t_min
                    fig_name = f"Master_Pred_Phase_{phase_idx}_at_Epoch_{epoch}_time_interval_tmin_{t_min:.5f}_tmax_{t_max:.5f}_{N_batches}_batches_t_min.png"
                else : 
                    fig_name = f"Master_Pred_Phase_{phase_idx}_at_Epoch_{epoch}_time_interval_tmin_{t_min:.5f}_tmax_{t_max:.5f}_{N_batches}_batches_t_max.png"
            plt.savefig(os.path.join(pathOutput ,fig_name)) 
            plt.close()

    ###############################################
    def process_repository_files_continous(self,path,pathOutput,title,filename):
        weights_files= self.PRE_POST.read_weights_files(path)
        weights_file=weights_files[0]
        #print(weights_file)
        #tf.print(weights_files)
        t_min = float('inf')
        t_max = float('-inf')

        # Make copy from self for testing and saving actual results        
        PINN_ = Sequentialmodel(layers=self.layers, X_f_train=self.X_f, X_ini_train=self.X_ini,\
                                phases_ini_indexes=self.phases_ini_indexes,all_ini_flags_matrix=self.All_flag_ini,\
                                Phi_ini=self.All_phi_ini,phi_ini_train=self.phi_ini, N_ini=self.N_ini,X_phi_test=self.X_phi_test,\
                                X_ini_train_all=self.X_ini_train_all, phi_ini_train_all=self.phi_ini_train_all,\
                                    all_interfaces=self.All_interfaces_ini,\
                                X_lb_train=self.X_lb, X_ub_train=self.X_rtb,\
                                X_ltb_train=self.X_ltb, X_rtb_train=self.X_rtb,\
                                X=None,Y=None,T=None,x=self.x,y=self.y,lb=self.lb, ub=self.ub, mu=self.mu, sigma=self.sigma, delta_g=self.delta_g,\
                                    eta=self.eta,Nx=self.Nx,Ny=self.Ny,Nt=self.Nt,phi_sol=None,pinns =self.pinns,num_phases=self.num_phases,
                                N_batches=self.N_batches,\
                                min_batch_numbers=self.min_batch_numbers,\
                                    Nbr_f_pts_max_per_batch=self.Nbr_f_pts_max_per_batch,\
                                Nbr_f_pts_min_per_batch=self.Nbr_f_pts_min_per_batch,\
                                N_ini_max_per_batch=self.N_ini_max_per_batch,\
                                N_ini_min_per_batch=self.N_ini_min_per_batch)      
        
        phi_evolution = []
        num_train_intervals=self.num_train_intervals
        time_subdomains=np.linspace(lb[2],ub[2],num_train_intervals+1)
        for i in range(num_train_intervals):
            t_min, t_max = time_subdomains[i],time_subdomains[i+1]
            #print(t_min,t_max)
            weights_loaded = self.PRE_POST.load_weights(weights_file)
            PINN_.set_weights(weights_loaded)

            X_phi_test_sub = PINN_.X_phi_test[:, 2] >= t_min
            X_phi_test_sub &= PINN_.X_phi_test[:, 2] <= t_max
            X_phi_test_sub = PINN_.X_phi_test[X_phi_test_sub, :]

            phi_pred = PINN_.evaluate(X_phi_test_sub).numpy()
            phi_pred = tf.reduce_sum(phi_pred, axis=0)
            
            phi_pred = np.reshape(phi_pred, (PINN_.Nx, PINN_.Ny, -1))
            phi_evolution.append(phi_pred[:, :, 0])
        #phi_evolution.append(phi_pred[:, :, -1])       
        
        num_boxes = 4     
        self.PRE_POST.plot_global_evolution_continous(num_boxes,phi_evolution, pathOutput,title,filename,t_max )
        
        del PINN_,phi_evolution,X_phi_test_sub,phi_pred

    ###############################################
    def save_predictions_discret_workers_Master(self,epoch,pathOutput,path_weights_all_pinns,X_phi_test,\
                                X_ini,u_ini,N_b,t_min, t_max,N_batches,flag_Master=False): 
        title = f"φ predicted for epoch_{epoch+1} - t_min: {t_min:.5f}, t_max: {t_max:.5f}" # should be epoch+1 (this for debug purpose)
        filename = f"phi_pred_epoch_{epoch+1} - t_min: {t_min:.5f}, t_max: {t_max:.5f}.jpg"
        tf.print("\n !!!  saving predictions of pinns in progress !!! \n")             
        path_weights = "weights/"
        self.process_repository_files_discret_workers_Master(epoch,N_batches,path_weights,pathOutput,path_weights_all_pinns,title,filename,flag_Master)
        tf.print("\n        ! saving complete ! \n")
    ###############################################
    def save_predictions_continous(self,epoch,pathOutput,X_phi_test,\
                                X_ini,u_ini,N_b,t_min, t_max): 
        title = f"phi_predicted by PINN for epoch_{epoch}" # should be epoch+1 (this for debug purpose)
        filename = f"phi_predicted by PINN for epoch_{epoch}.jpg"
        
        path_weights = "weights/"
        self.process_repository_files_continous(path_weights,pathOutput,title,filename)
    ###############################################
    def save_predictions_regular_int(self,epoch,pathOutput,X_phi_test,\
                                X_ini,u_ini,N_b,t_min, t_max):
        X_phi_test_sub = X_phi_test[:, 2] >= t_min
        X_phi_test_sub &= X_phi_test[:, 2] <= t_max
        X_phi_test_sub = X_phi_test[X_phi_test_sub, :]
        phi_pred = self.evaluate(X_phi_test_sub)
        phi_pred = np.reshape(phi_pred,(self.Nx,self.Ny,-1))  
        num_boxes = 3   # Time intervals

        box_size = phi_pred.shape[2] // num_boxes

        # Compute the number of rows needed to display all subplots
        num_rows = (num_boxes + 1) // 2

        # Create the figure and subplots
        fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(8, 4*num_rows))

        # Loop over the boxes and plot the corresponding phi values
        out_area_vs_t=[]
        out_radius_vs_t=[]

        thresh=1e-3

        for i, ax in enumerate(axes.flat):
            if i < num_boxes :
                start_idx = i * box_size
                end_idx = (i + 1) * box_size
                phi=phi_pred[:, :, start_idx]
                phi = np.clip(phi, 0, 1)
                im=ax.imshow(phi, cmap='jet', interpolation='none', vmin=0, vmax=1)
                cbar = fig.colorbar(im, ax=ax, shrink=0.5)
                cbar.ax.set_ylabel(r'$\phi$')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_title(f'Time: {start_idx}')
                t=i * box_size
                area_vs_t=(len(phi[phi>thresh]))/100
                
            else:
                for j in range (phi_pred.shape[2]):
                    if j % 10==0:
                        phi=phi_pred[:, :, j]
                        t=j
                        area_vs_t=(len(phi[phi>thresh]))
                        out_area_vs_t.append([t,area_vs_t])
                        
                out_radius_vs_t=np.sqrt(np.asarray(out_area_vs_t)[:,1] /np.pi)
                out_time=np.asarray(out_area_vs_t)[:,0] 
                # Add the area vs. time plot to the last subplot
                ax.plot(out_time[::], out_radius_vs_t[::], "r",label=r"$PINN$",linestyle=' ',marker='o')
                ax.set_xlabel('Time')
                ax.set_ylabel('Radius')
                ax.set_title('Radius vs. Time')
                break

        title = f"φ predicted for epoch_{epoch+1} - t_min: {t_min:.5f}, t_max: {t_max:.5f}" # should be epoch+1 (this for debug purpose)
        fig.suptitle(title)
      
        filename = f"phi_pred_epoch_{epoch+1} - t_min: {t_min:.5f}, t_max: {t_max:.5f}.jpg"
        plt.savefig(os.path.join(pathOutput ,filename))
        plt.close()
    ###############################################
    def save_weights(self, PINN,weights_file):
        lock= multiprocessing.Lock()
        lock.acquire()
        weights_dict={}
        weights_key=0
        weights_dict[weights_key] = {}
        weights_dict[weights_key]['t_min'] = PINN.t_min
        weights_dict[weights_key]['t_max'] = PINN.t_max
        weights_loaded=tf.cast(PINN.get_weights(), dtype=tf.float64)
        weights_dict[weights_key]['weights'] = [w.numpy() for w in weights_loaded]
        #tf.print(PINN.idx_pinn)
        with open(weights_file, 'w') as f:
            json.dump(weights_dict[weights_key], f, cls=NumpyEncoder)
        lock.release()
    ###############################################
    def set_weights_Master_PINN(self):
        weights_files = glob.glob('get_weights/*.json')
        weights_files = sorted(weights_files)
        weights_file = weights_files[0]
        match = re.search(r'_tmin_(\d+\.\d+)_tmax_(\d+\.\d+)_', weights_file)
        if match:
            t_min = float(match.group(1))
            t_max = float(match.group(2))
            #print(t_min, t_max)
        else:
            print("Unable to extract t_min and t_max from the filename.")
            
        with open(weights_file, 'r') as f:
            weights_loaded =json.load(f)['weights']
        weights_loaded=tf.cast(weights_loaded, dtype=self.precision)
        self.set_weights(weights_loaded) 
        #print("PINN_ weights initialized")
    ##############################################
    def update_selection_for_training_domain_und_boundaries(self,X_f_sub_domain):
        t_min=self.t_min
        t_max=self.t_max
        X_lb_sub_domain  = self.X_lb[np.logical_and(t_min <= self.X_lb[:,2], self.X_lb[:,2] <= t_max)]
        X_ub_sub_domain  = self.X_ub[np.logical_and(t_min <= self.X_ub[:,2], self.X_ub[:,2] <= t_max)]
        X_ltb_sub_domain  = self.X_ltb[np.logical_and(t_min <= self.X_ltb[:,2], self.X_ltb[:,2] <= t_max)]
        X_rtb_sub_domain  = self.X_rtb[np.logical_and(t_min <= self.X_rtb[:,2], self.X_rtb[:,2] <= t_max)]
        X_lb_sub_domain[:, 0] = np.linspace(start=self.lb[2], stop=self.ub[2], num=X_lb_sub_domain.shape[0], endpoint=True)
        X_ub_sub_domain[:, 0] = np.linspace(start=self.lb[2], stop=self.ub[2], num=X_ub_sub_domain.shape[0], endpoint=True)
        X_ltb_sub_domain[:, 1] = np.linspace(start=self.lb[2], stop=self.ub[2], num=X_ltb_sub_domain.shape[0], endpoint=True)
        X_rtb_sub_domain[:, 1] = np.linspace(start=self.lb[2], stop=self.ub[2], num=X_rtb_sub_domain.shape[0], endpoint=True)
        abs_x_min  = tf.reduce_min(X_lb_sub_domain[:,0])
        abs_x_max  = tf.reduce_max(X_lb_sub_domain[:,0])
        abs_y_min  = tf.reduce_min(X_ltb_sub_domain[:,1])
        abs_y_max  = tf.reduce_max(X_ltb_sub_domain[:,1])
        self.abs_x_min=abs_x_min
        self.abs_x_max=abs_x_max
        self.abs_y_min =abs_y_min
        self.abs_y_max=abs_y_max
        self.X_f_sub_domain = X_f_sub_domain             
        self.X_lb_sub_domain = X_lb_sub_domain
        self.X_ub_sub_domain = X_ub_sub_domain
        self.X_ltb_sub_domain = X_ltb_sub_domain
        self.X_rtb_sub_domain = X_rtb_sub_domain
    ############################################## 
    def initialize_dict_I(self,N_batches) : 
      num_phases = self.num_phases  
      # The above code is declaring a variable named `dict_I` in Python. However, it is not assigning
      # any value to the variable, so it is currently empty.
      dict_I = {}
      for batch_idx in range(N_batches):
          for phase_idx in range(num_phases):
              batch_key = f"batch_{batch_idx}"
              phase_key = f"phase_{phase_idx}"
              dict_I[f"{batch_key}_{phase_key}_value"] = 0
      return dict_I
    ##############################################              
    def update_interaction(self,interactions, key, new_value):
        phase_idx = key[0]
        batch_idx = key[1]
        alpha = key[2]
        beta = key[3]
        phase_label = f"Phase_{phase_idx}"
        batch_label = f"Batch_{batch_idx}"
        key = (phase_label, batch_label, alpha, beta)
        str_key = str(key)
        if str_key in interactions:
            interactions[str_key] = new_value
            return True
        else:
            print(f"Interaction not found for key: {str_key}")
            return False
    ##############################################
    def get_interaction_value(self, key):
        global interactions
        phase_idx = key[0]
        batch_idx = key[1]
        alpha = key[2]
        beta = key[3]
        phase_label = f"Phase_{phase_idx}"
        batch_label = f"Batch_{batch_idx}"
        key = (phase_label, batch_label, alpha, beta)
        str_key = str(key)
        if str_key in interactions:
            value=interactions[str_key] 
            return value
        else:
            print(f"Interaction not found for key: {str_key}")
            return False
   ##############################################              
    def plot_interaction(self,interactions, N_batches):    
        num_phases = self.num_phases
        num_batches = N_batches

        interaction_matrix = np.zeros((num_phases, num_phases, num_batches))
        
        # Populate the interaction matrix based on your interactions data
        for key, value in interactions.items():
            key_tuple = eval(key)
        
            phase_i = int(key_tuple[2])
            phase_j = int(key_tuple[3])
            batch_idx =int(key_tuple[1].split('_')[1])
            interaction_matrix[phase_i][phase_j][batch_idx] = value
        phases = sorted(set(eval(key)[0] for key in interactions))
        batches = sorted(set(eval(key)[1] for key in interactions))
        
        num_phases = len(phases)
        num_batches = len(batches)
        
        # Create subplots for each batch
        fig, axes = plt.subplots(1, num_batches, figsize=(4 * num_batches, 4))
        
        # Create heatmaps for each batch
        for batch_idx, batch_name in enumerate(batches):
            interaction_matrix_2D = np.zeros((num_phases, num_phases))
            
            for phase_i, phase_name_i in enumerate(phases):
                for phase_j, phase_name_j in enumerate(phases):
                    key = (phase_name_i, batch_name, phase_i, phase_j)
                    if str(key) in interactions:  # Check if the key exists in interactions
                        value=interactions[str(key)]
                        interaction_matrix_2D[phase_i][phase_j] = value
            
            sns.heatmap(
                interaction_matrix_2D,
                ax=axes[batch_idx],
                annot=True,
                cmap='coolwarm',
                fmt='.0f',
                cbar=True,
                xticklabels=[f'Phase_{i}' for i in range(num_phases)],
                yticklabels=[f'Phase_{i}' for i in range(num_phases)]
            )
            axes[batch_idx].set_title(f'Batch {batch_name}')
        
        plt.tight_layout()
        global epoch 
        title=f"intercations_Heatmap_Epoch {epoch}_Time interval: t_min: {self.t_min:.5f}, t_max: {self.t_max:.5f}N_batches_{N_batches}.jpg"
        if N_batches==4:
          plt.savefig(os.path.join("save_figs",title), dpi = 500, bbox_inches='tight') 
        plt.close()
    ##############################################              
    def set_phases_interaction(self, N_batches, interactions):
        # Define a proximity threshold for interaction (adjust as needed)
        interaction_threshold = 0.001
        count_no_interaction = 0
        num_phases=self.num_phases
        thresh_interface= self.thresh_interface
        
        for idx_phase_alpha in range(num_phases):
        
            for idx_batch in range(N_batches):
                pinn_alpha_idx_batch = str(idx_batch).zfill(2)
                pinn_alpha_idx_pinn = str(idx_batch).zfill(2)  + str(idx_phase_alpha).zfill(2) 
                pinn_alpha_idx_phase = str(idx_phase_alpha).zfill(2)
                pinn_alpha=self.find_pinn_by_idx_pinn(pinn_alpha_idx_pinn, target_pinns=None)  # for memory save, call global_pinns inside function
                X_Y_alpha = pinn_alpha.batch_X_ini[:, :2]
                tree_phase_0 = cKDTree(X_Y_alpha)
        
                # Loop through all phases except the reference phase 
                for idx_phase_beta in range(num_phases):
                    if idx_phase_alpha != idx_phase_beta:
                        pinn_beta_idx_pinn =str(idx_batch).zfill(2)  + str(idx_phase_beta).zfill(2)
                        pinn_beta=self.find_pinn_by_idx_pinn(pinn_beta_idx_pinn, target_pinns=None)
                        X_Y_beta= pinn_beta.batch_X_ini[:, :2]
        
                        # Create KD-tree for the current phase
                        tree_phase_1 = cKDTree(X_Y_beta)
        
                        # Find pairs of points from different phases that are within the interaction threshold
                        pairs = tree_phase_0.query_ball_tree(tree_phase_1, r=interaction_threshold)
        
                        # Extract the interacting points using a list comprehension
                        interacting_points_phase_0 = [X_Y_alpha[i] for i, pair in enumerate(pairs) if pair]
                        interacting_points_phase_1 = [X_Y_beta[j] for j in {j for i in pairs for j in i}]
                        key = [idx_phase_alpha, idx_batch, idx_phase_alpha, idx_phase_beta]
        
                        if interacting_points_phase_0:
                            self.update_interaction(interactions, key, new_value=1)
                        else:
                            self.update_interaction(interactions, key, new_value=0)
                            count_no_interaction += 1
        return interactions, count_no_interaction
    ##############################################
    def Phases_interactions_infos(self,N_batches):
        num_phases = self.num_phases
        interactions = {}
        phase_labels = [f"Phase_{i}" for i in range(num_phases)]
        batch_labels = [f"Batch_{i}" for i in range(N_batches)]
        total_interactions=0
        for phase_idx in range(num_phases):
            for batch_idx in range(N_batches):
                for alpha in range(num_phases):
                    for beta in range(num_phases):
                        if alpha != beta and alpha==phase_idx:
                            phase_alpha = phase_labels[alpha]
                            phase_beta = phase_labels[beta]
                            key = (phase_labels[phase_idx], batch_labels[batch_idx], alpha, beta)
                            str_key = str(key)
                            interactions[str_key] = 1
                            total_interactions+=1
    
        directory_path = "Dictionnaries"
        json_file_name = f"Phases_interactions_infos_{N_batches}_batches.json"
        json_file_path = os.path.join(os.getcwd(), directory_path, json_file_name)
    
        # Ensure the directory exists, create it if necessary
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
    
        # Store dict as JSON
        with open(json_file_path, 'w') as json_file:
            json.dump(interactions, json_file)
    
        return interactions, total_interactions
    ##############################################            
    def get_and_store_quarters_max_indices(self,pinns,quarters,N_batches):  
        info_dict = {}  #  store the information about the selected pinn for the next training Etage
        quarters_max_indices = []
        batches_indices_ref=np.zeros(len(quarters) )
        for pinn_id in range(self.num_phases): 
            quarters_info = {} 
            for quarter_number, (quarter_name, batch_indices) in enumerate(quarters.items()):
                max_int_percentage_per_quarter = -1
                max_pinn_index = None
                max_pinn_infos = []    
                
                for batch_idx in batch_indices:
                    pinn_index = batch_idx * self.num_phases + pinn_id
                    pinn = self.pinns[pinn_index]
                    idx_batch = pinn.idx_batch
                    idx_pinn = pinn.idx_pinn
                    #tf.print(f"Pinn ID: {pinn_id}, Quarter: {quarter_name}, Batch: {idx_batch}, PINN: {idx_pinn}")
                      
                    if pinn_id==0:  
                      
                      per_interface_points = pinn.percentage_inter_points
                      if per_interface_points > max_int_percentage_per_quarter:
                          max_int_percentage_per_quarter = per_interface_points
                          max_pinn_index = idx_pinn   
                          max_pinn_info = {
                          "idx_pinn": idx_pinn,
                          "limits": pinn.limits
                                          }
                          batches_indices_ref[quarter_number]=idx_batch  
                    else:  
                         if idx_batch==batches_indices_ref[quarter_number]  :                                                          
                            max_pinn_index = idx_pinn   
                            max_pinn_info = {
                            "idx_pinn": idx_pinn,
                            "limits": pinn.limits
                                            }                                          
                                  
                max_pinn_infos.append(max_pinn_info)
                quarters_max_indices.append(max_pinn_index)
                quarters_info[quarter_name] = max_pinn_infos 
            #tf.print("ref ", batches_indices_ref ) 
            phase_key = f"Phase_{pinn_id}"
            info_dict[phase_key] = quarters_info
            
        json_file_name = f"quarters_infos_{N_batches}_batches.json"
        json_file_path = os.path.join(os.path.join(os.getcwd(),"Dictionnaries"),json_file_name)
        # store dict 
        with open(json_file_path, 'w') as json_file:
            json.dump(info_dict, json_file)

        return quarters_max_indices, info_dict
  
    ##############################################
    def get_and_store_quarters_max_indices_case_2(self, quarters, N_batches, old_info_dict):  
        # Make a copy of the old_info_dict
        new_info_dict = old_info_dict.copy()
        
        quarters_max_indices = []
        batches_indices_ref=np.zeros(len(quarters) )
        for pinn_id in range(self.num_phases): 
            quarters_info = {} 
            
            for quarter_number, (quarter_name, batch_indices) in enumerate(quarters.items()):
                max_pinn_index = None
                max_pinn_infos = []     
                
                for batch_idx in batch_indices:
                    pinn_index = batch_idx * self.num_phases + pinn_id
                    pinn = self.pinns[pinn_index]
                    idx_batch = pinn.idx_batch
                    idx_pinn = pinn.idx_pinn
                    idx_phase = pinn.idx_phase
                    
                    # Define phase_key
                    phase_key = f"Phase_{pinn_id}"
                    
                    if int(idx_phase)==0:
                      # Check if phase_key and quarter_name exist in old_info_dict
                      if phase_key in old_info_dict and quarter_name in old_info_dict[phase_key]:
                          # Get the information for the old PINN
                          pinn_info = old_info_dict[phase_key][quarter_name]
                          #tf.print("pinn_info : ", pinn_info)
                          idx_pinn_old = pinn_info[0]["idx_pinn"]
                          limits_old = pinn_info[0]["limits"]
                          
                          # Check if there is an intersection with the old PINN
                          if self.is_intersection(pinn.limits, limits_old):
                              max_pinn_index = idx_pinn
                              max_pinn_info = {
                                  "idx_pinn": idx_pinn,
                                  "limits": pinn.limits
                              }
                              max_pinn_infos.append(max_pinn_info)
                              batches_indices_ref[quarter_number]=idx_batch
                              break 
                          
                    else:  
                        if idx_batch==batches_indices_ref[quarter_number]  :                                                          
                            max_pinn_index = idx_pinn   
                            max_pinn_info = {
                            "idx_pinn": idx_pinn,
                            "limits": pinn.limits
                                          }                              
                
                # Append the selected PINN info to quarters_info                
                max_pinn_infos.append(max_pinn_info)
                quarters_max_indices.append(max_pinn_index)
                quarters_info[quarter_name] = max_pinn_infos 
                
  
            
            # Add quarters_info to new_info_dict for the current phase
            phase_key = f"Phase_{pinn_id}"
            new_info_dict[phase_key] = quarters_info
   
        json_file_name = f"quarters_infos_{N_batches}_batches.json"
        json_file_path = os.path.join(os.path.join(os.getcwd(),"Dictionnaries"),json_file_name)
        # store dict 
        with open(json_file_path, 'w') as json_file:
            json.dump(new_info_dict, json_file)
        #tf.print("new_info_dict: ", new_info_dict)

        return quarters_max_indices, new_info_dict            
    ##############################################
    def get_all_indices_sampled(self,pinn,fraction_ones_per_int_pts,fraction_zeros_per_int_pts): 
        # here we construct an array in which first we put only interfacial points, then we add to it fraction_ones_per_int_pts % ONE points 
        # and fraction_zeros_per_int_pts % zeros points
        row_sums = pinn.phi_ini_all_sub_domain
        thresh_interface=pinn.thresh_interface
        interfacial_indices = np.where(np.logical_and(row_sums > thresh_interface, row_sums < 1-thresh_interface))[0]
        interfacial_indices_sampled = np.random.choice(interfacial_indices, size=int(len(interfacial_indices) ), replace=False)
        zero_indices = np.where(row_sums <= thresh_interface)[0]

        one_indices = np.where(row_sums >= 1 - thresh_interface)[0]
        
        if len(zero_indices>0):
            replace_0= False if len(zero_indices) > int(len(interfacial_indices_sampled) * fraction_zeros_per_int_pts) else True
            zero_indices_sampled = np.random.choice(zero_indices, size=int(len(interfacial_indices_sampled) * fraction_zeros_per_int_pts), replace=False)
        
        if len(one_indices>0):
            replace_1=False if len(one_indices) > int(len(interfacial_indices_sampled) * fraction_ones_per_int_pts) else True
            one_indices_sampled = np.random.choice(one_indices, size=int(len(interfacial_indices_sampled) * fraction_ones_per_int_pts), replace=False)

        all_indices_sampled = np.concatenate([interfacial_indices_sampled, one_indices_sampled,zero_indices_sampled])
        return all_indices_sampled
    ##############################################
    def get_all_indices_sampled_2(self,pinn,fraction_ones_per_int_pts,fraction_zeros_per_int_pts,N_ini_min_per_batch,N_ini_max_per_batch): 
        # here we construct an array in which first we put only interfacial points, then we add to it fraction_ones_per_int_pts % ONE points 
        # and fraction_zeros_per_int_pts % zeros points
        phi = pinn.phi_ini_all_sub_domain
        
        thresh_interface=pinn.thresh_interface
        interfacial_indices = np.where(np.logical_and(phi >= thresh_interface, phi <= 1-thresh_interface))[0]
        pinn.percentage_inter_points=len(interfacial_indices)/len(phi)
        pinn.flag=1 if len(interfacial_indices)>0 else 0
        
        # if no interfacial points, pinn will not be considered in training (just take few points for plot and minor check)
        if pinn.flag==0:  
            random_indices= np.random.choice(len(phi), size=min(1,N_ini_min_per_batch) , replace=False)
            return random_indices
        
        # pinn.flag=1
        #interfacial_indices_sampled = np.random.choice(interfacial_indices, size=int(len(interfacial_indices) ), replace=False)
        zero_indices = np.where(phi <= thresh_interface)[0]
        one_indices = np.where(phi >= 1 - thresh_interface)[0]
        
        def sigmoid(x,N_ini_min_batch,N_ini_max_batch):
            return N_ini_min_batch + (N_ini_max_batch - N_ini_min_batch) / (1 + math.exp(-x))
        
        def get_N_ini(replace_0,coef=1):
            zero_indices_reduced = np.random.choice(zero_indices, size=int(len(interfacial_indices) * fraction_zeros_per_int_pts*coef), replace=replace_0)
            interfacial_indices_reduced = np.random.choice(interfacial_indices, size=int(len(interfacial_indices)-len(zero_indices_reduced) ), replace=False)
            interfacial_zero_indices_combined = np.concatenate([interfacial_indices_reduced, zero_indices_reduced])
            N_ini_per_batch = int(pinn.percentage_inter_points *(N_ini_max_per_batch-N_ini_min_per_batch) + N_ini_min_per_batch)
            x = pinn.percentage_inter_points * 100 # adjust the scaling factor 
            N_ini_per_batch = int(sigmoid(x,N_ini_min_per_batch,N_ini_max_per_batch))
            N_ini_per_batch = min(N_ini_per_batch, N_ini_max_per_batch)
            return N_ini_per_batch, interfacial_zero_indices_combined
            
        def get_interfacial_indices_sampled(pinn,interfacial_indices,zero_indices,fraction_zeros_per_int_pts):
            replace_0= False if len(zero_indices) > int(len(interfacial_indices) * fraction_zeros_per_int_pts)  else True
            N_ini_per_batch, interfacial_zero_indices_combined=get_N_ini(replace_0)
    
            replace_1= False if len(interfacial_zero_indices_combined) > N_ini_per_batch  else True
            interfacial_indices_sampled =np.random.choice(interfacial_zero_indices_combined, size=N_ini_per_batch, replace=replace_1)
            return interfacial_indices_sampled
        
        def get_interfacial_indices_sampled_only_int(pinn,interfacial_indices):
            interfacial_zero_indices_combined = interfacial_zero_indices_combined
            N_ini_per_batch = int(pinn.percentage_inter_points *(N_ini_max_per_batch-N_ini_min_per_batch) + N_ini_min_per_batch)
            x = pinn.percentage_inter_points * 100 # adjust the scaling factor 
            N_ini_per_batch = int(sigmoid(x,N_ini_min_per_batch,N_ini_max_per_batch))
            N_ini_per_batch = min(N_ini_per_batch, N_ini_max_per_batch)
            replace_1= False if len(interfacial_zero_indices_combined) > N_ini_per_batch  else True
            interfacial_indices_sampled =np.random.choice(interfacial_zero_indices_combined, size=N_ini_per_batch, replace=replace_1)
            #tf.print("here ",pinn.idx_pinn, len(interfacial_indices_sampled))
            return interfacial_indices_sampled     

        if len(zero_indices)>0 and len(one_indices)==0:
            return get_interfacial_indices_sampled(pinn,interfacial_indices,zero_indices,fraction_zeros_per_int_pts)
        
        if len(zero_indices)==0 and len(one_indices)==0:
            return get_interfacial_indices_sampled_only_int(pinn,interfacial_indices)
            
        if len(zero_indices)==0 and len(one_indices)>0:
            return get_interfacial_indices_sampled(pinn,interfacial_indices,one_indices,fraction_ones_per_int_pts)
        
        if len(zero_indices)>0 and len(one_indices)>0:
            replace_0 = False if len(zero_indices) > int(len(interfacial_indices) * fraction_zeros_per_int_pts) else True
            coef_0 = 1 if len(zero_indices) > int(len(interfacial_indices) * fraction_zeros_per_int_pts) else 1
            zero_indices_reduced = np.random.choice(zero_indices, size=int(len(interfacial_indices) * fraction_zeros_per_int_pts * coef_0), replace=replace_0)

            replace_1= False if len(one_indices) > int(len(interfacial_indices) * fraction_ones_per_int_pts)  else True
            coef_1 = 1 if len(one_indices) > int(len(interfacial_indices) * fraction_ones_per_int_pts) else 1
            one_indices_reduced = np.random.choice(one_indices, size=int(len(interfacial_indices) * fraction_ones_per_int_pts* coef_1), replace=replace_1)

            interfacial_indices_reduced = np.random.choice(interfacial_indices, size=int(len(interfacial_indices)-len(zero_indices_reduced)-len(one_indices_reduced) ), replace=False)            
            interfacial_zero_ones_indices_combined = np.concatenate([interfacial_indices_reduced, zero_indices_reduced,one_indices_reduced])            
            x = pinn.percentage_inter_points * 100
            N_ini_per_batch = int(sigmoid(x,N_ini_min_per_batch,N_ini_max_per_batch))# int(pinn.percentage_inter_points *(N_ini_max_per_batch-N_ini_min_per_batch) + N_ini_min_per_batch)
            N_ini_per_batch = min(N_ini_per_batch, N_ini_max_per_batch)
            replace_= False if len(interfacial_zero_ones_indices_combined) > N_ini_per_batch  else True
            interfacial_indices_sampled =np.random.choice(interfacial_zero_ones_indices_combined, size=N_ini_per_batch, replace=replace_)            
        
        return interfacial_indices_sampled
    
    ##############################################   
    ############################################## 
    def extract_near_interface_indices(self, pinn, X, phi, threshold):
      thresh_interface = self.thresh_interface
  
      # Identify points within the interface
      interface_indices = np.where(np.logical_and(phi > thresh_interface, phi < 1 - thresh_interface))[0]
  
      near_interface_indices = []
      X_interface = X[interface_indices, :2]
  
      # Calculate the distance matrix between all points in X and interface points
      distances = np.linalg.norm(X[:, :2][:, np.newaxis] - X_interface, axis=2)
  
      # Find points near the interface within the specified threshold
      near_indices = np.where(distances < threshold)
      near_interface_indices = near_indices[0]
  
      return np.unique(near_interface_indices)

     ############################################## 
    ##############################################  
    def extract_far_from_interface_indices(self, X, phi, threshold):
        thresh_interface = self.thresh_interface
        #  points within the interface
        interface_indices = np.where(np.logical_and(phi > thresh_interface, phi < 1 - thresh_interface))[0]
    
        #  the distance matrix between all points in X and interface points
        interface_coords = X[interface_indices]
        distances = np.linalg.norm(X[:, np.newaxis, :] - interface_coords, axis=2)

        is_far = np.all(distances >= threshold, axis=1)
    
        far_from_interface_indices = np.where(is_far)[0]
    
        return far_from_interface_indices, interface_indices
    ##############################################   
    def set_IC_data_for_pinns(self,N_batches,fraction_ones_per_int_pts,fraction_zeros_per_int_pts,\
        N_ini_min_per_batch,N_ini_max_per_batch, flag_plot=True):
        thresh_interface=self.thresh_interface
        # get for each pinn initial data (IC)
        num_x_intervals = int(np.sqrt(N_batches))
        num_y_intervals = int(np.sqrt(N_batches))
        step_x = max(self.x) / num_x_intervals
        step_y = max(self.y) / num_x_intervals  
        t_min=self.t_min
        t_max=self.t_max
        global epoch
        global path_save
        
        X_ini_all_sub_domain_= []                   
             
        #tf.print(self.X_ini_all_sub_domain.shape, self.phi_ini_all_sub_domain.shape)
        for phase_idx in range(self.num_phases): #this is a plot loop, no computng associated with 
            IC_nbr=0
            fig, ax = plt.subplots()                 
            for counter, pinn in enumerate(self.pinns):
                pinn.flag_one_side=0 
                pinn.flag_grain=0
                pinn.flag_no_grain=0
                batch_idx = counter // self.num_phases
                pinn_id = counter % self.num_phases
                n=len(self.X_ini_all_sub_domain)
                N=self.num_phases
                pinn.X_ini_all_sub_domain= self.X_ini_all_sub_domain[int(n*pinn_id/N)+1:int(n*(pinn_id+1)/N)]
                pinn.phi_ini_all_sub_domain= self.phi_ini_all_sub_domain[int(n*pinn_id/N)+1:int(n*(pinn_id+1)/N)]

                i = batch_idx  // num_x_intervals
                j = batch_idx %  num_y_intervals
                x_min = i * step_x
                x_max = (i + 1) * step_x
                y_min = j * step_y
                y_max = (j + 1) * step_y


                batch_indices = np.where((pinn.X_ini_all_sub_domain[:, 0] >= x_min) & (pinn.X_ini_all_sub_domain[:, 0] <= x_max) &
                                    (pinn.X_ini_all_sub_domain[:, 1] >= y_min) & (pinn.X_ini_all_sub_domain[:, 1] <= y_max))[0]
                pinn.all_ini_indices=batch_indices
                
                pinn.X_ini_all_sub_domain=pinn.X_ini_all_sub_domain[pinn.all_ini_indices]
                pinn.X_ini_all_sub_domain[:,2]=t_min
                if t_min==0:
                    pinn.phi_ini_all_sub_domain=pinn.phi_ini_all_sub_domain[pinn.all_ini_indices] 
                else:
                    pinn.phi_ini_all_sub_domain=np.asarray(pinn.evaluate(pinn.X_ini_all_sub_domain))
                #f.print("chck shapes", pinn.X_ini_all_sub_domain.shape, pinn.phi_ini_all_sub_domain.shape)
                                  
                #"""
                # near interface indices (interfaces + near points grouped )
                near_interface_indices = self.extract_near_interface_indices(pinn,pinn.X_ini_all_sub_domain[:,:2], pinn.phi_ini_all_sub_domain, self.eta/2) 
                # far from interface indices (only far points grouped )
                far_from_interface_indices,interface_indices = self.extract_far_from_interface_indices(pinn.X_ini_all_sub_domain[:,:2], pinn.phi_ini_all_sub_domain, self.eta/2)
                # first extract the far bacth
                pinn.far_int_batch_X_ini=pinn.X_ini_all_sub_domain[far_from_interface_indices]
                pinn.far_int_batch_phi_ini=pinn.phi_ini_all_sub_domain[far_from_interface_indices]   

                # then group the (interfaces + near points ) together in one batch (if percentage_zeros==percentage_ones=0 ==> only Interface considered )
                #pinn.phi_ini_all_sub_domain=pinn.phi_ini_all_sub_domain[near_interface_indices]
                #pinn.X_ini_all_sub_domain=pinn.X_ini_all_sub_domain[near_interface_indices]                              
                #"""

                all_indices_sampled=self.get_all_indices_sampled_2(pinn,fraction_ones_per_int_pts,fraction_zeros_per_int_pts,N_ini_min_per_batch,N_ini_max_per_batch)  
                pinn.X_ini_all_sub_domain_reduced=pinn.X_ini_all_sub_domain[all_indices_sampled]
                pinn.phi_ini_all_sub_domain_reduced=pinn.phi_ini_all_sub_domain[all_indices_sampled]
                global denoising_loss_
                if denoising_loss_ ==True:
                    # far from the interface batch 
                    N_samples_far_int = max(len(all_indices_sampled) // 5,16)  # internal code management 
                    group1_indices = np.where(pinn.far_int_batch_phi_ini >= (1 - pinn.thresh_interface))[0] # far from the interface : grain
                    group2_indices =  np.where(pinn.far_int_batch_phi_ini <= pinn.thresh_interface)[0]       # far from the interface : no-grain
                    if len(group1_indices) == 0 or len(group2_indices) == 0:
                        pinn.flag_one_side=1  
                        if len(group1_indices) == 0:
                            #tf.print(group2_indices)
                            # when group1 is empty
                            pinn.flag_no_grain=1
                            pinn.flag_grain=0
                            selected_indices_group2 = np.random.choice(group2_indices, N_samples_far_int, replace=False)
                            pinn.far_int_batch_X_ini = pinn.far_int_batch_X_ini[selected_indices_group2]
                            pinn.far_int_batch_phi_ini = pinn.far_int_batch_phi_ini[selected_indices_group2]
                    
                        else:
                            # when group2 is empty
                            pinn.flag_grain=1
                            pinn.flag_no_grain=0
                            selected_indices_group1 = np.random.choice(group1_indices, N_samples_far_int, replace=False)
                            pinn.far_int_batch_X_ini = pinn.far_int_batch_X_ini[selected_indices_group1]
                            pinn.far_int_batch_phi_ini = pinn.far_int_batch_phi_ini[selected_indices_group1]

                    else:
                        # sample points from both groups
                        if len(group1_indices) > N_samples_far_int:
                            selected_indices_group1 = np.random.choice(group1_indices, N_samples_far_int, replace=False)
                        else:
                            selected_indices_group1 = group1_indices
                    
                        if len(group2_indices) > N_samples_far_int:
                            selected_indices_group2 = np.random.choice(group2_indices, N_samples_far_int, replace=False)
                        else:
                            selected_indices_group2 = group2_indices
                        min_length = min(len(selected_indices_group1), len(selected_indices_group2))
                        selected_indices_group1 = np.random.choice(selected_indices_group1, min_length, replace=False)
                        selected_indices_group2 = np.random.choice(selected_indices_group2, min_length, replace=False)
                        selected_indices = np.concatenate((selected_indices_group1, selected_indices_group2))

                        pinn.far_int_batch_X_ini = pinn.far_int_batch_X_ini[selected_indices]
                        pinn.far_int_batch_phi_ini = pinn.far_int_batch_phi_ini[selected_indices]
                    
                    # update time column     
                    pinn.t_min=self.t_min
                    pinn.t_max=self.t_max
                    updated_time_column = np.random.uniform(self.t_min, self.t_max, pinn.far_int_batch_X_ini.shape[0])
                    pinn.far_int_batch_X_ini[:, 2]= updated_time_column
                    #tf.print("pinn.far_int_batch_X_ini ",pinn.far_int_batch_X_ini)
                   
                # Debug 
                """
                if pinn.idx_pinn=="0000":
                    tf.print("pinn.far_int_batch_X_ini ",pinn.far_int_batch_X_ini)
                    tf.print("pinn.t_min:", pinn.t_min)
                    tf.print("pinn.t_max:", pinn.t_max)
                    tf.print(np.random.uniform(self.t_min, self.t_max, pinn.far_int_batch_X_ini.shape[0]))
                    fig = plt.figure()
                    scatter=plt.scatter(pinn.far_int_batch_X_ini[:, 0], pinn.far_int_batch_X_ini[:, 1], cmap=plt.get_cmap('jet'), c=pinn.far_int_batch_phi_ini)
                    cbar = plt.colorbar(scatter, shrink=0.35)
                    cbar.set_label('phi')
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    tf.print(pinn.idx_pinn)
                    tf.print(len(far_from_interface_indices))
                    plt.savefig("far_int_batch_X_ini.png")  # Added the ".png" extension to the filename
                    tf.print(N_samples_far_int)
                    plt.close()
                    tf.print(stop_here)  
                """                
                    
                # update key self parameters 
                pinn.limits=[x_min,x_max,y_min,y_max]
                pinn.batch_X_ini=pinn.X_ini_all_sub_domain_reduced
                pinn.batch_phi_ini=pinn.phi_ini_all_sub_domain_reduced
                pinn.batch_indices=batch_indices
                pinn.loss_value=1
                self.pinns[pinn.order].loss_value=1
                # mem release
                #del pinn.X_ini_all_sub_domain_reduced, pinn.phi_ini_all_sub_domain_reduced,  pinn.X_ini_all_sub_domain, pinn.phi_ini_all_sub_domain

                if  pinn_id==phase_idx :
                    phi= pinn.batch_phi_ini
                    IC_nbr+=len(phi)
                    X_Y=pinn.batch_X_ini
                    x_avg = (x_min + x_max) / 2
                    L=abs(x_max-x_min)
                    W= abs(y_max-y_min)
                    y_avg = (y_min + y_max) / 2
                    color = np.random.rand(3)
                    ax.text(x_avg, y_avg, f"{pinn.idx_pinn}", color=color, ha='right', va='bottom',fontsize=10)
                    ax.text(x_avg-L/4, y_avg-W/4, f"{len(phi)} pts", color='orange', ha='right', va='bottom',fontsize=8)
                    ax.scatter(X_Y[:, 0], X_Y[:, 1], color=color, marker='o',s=25, label='IC')
                    scatter_IC=ax.scatter(X_Y[:, 0], X_Y[:, 1], cmap=plt.get_cmap('viridis'), c=phi,marker='o',s=10,vmin=0,vmax=1)
 
  
 
            title=f"IC_points_({IC_nbr})_at_Epoch_{epoch}_for_Time_interval_t_min_{pinn.t_min:.5f}_t_max_{pinn.t_max:.5f}_Phase_{phase_idx}.jpg"
            #if epoch==0 or flag==1 or flag_reduce_batches==1 and flag_shuffle_for_scipy==0:
            plt.grid(True)
            plt.xticks(np.linspace(self.x.min(), self.x.max(), num_x_intervals+1))
            plt.yticks(np.linspace(self.y.min(), self.y.max(), num_y_intervals+1))
            if flag_plot==True:
                plt.savefig(os.path.join(path_save,title), dpi = 500, bbox_inches='tight')             
            plt.close()

            tf.print("shapes: ",  self.X_ini_all_sub_domain.shape,self.phi_ini_all_sub_domain.shape)

            return num_x_intervals, num_y_intervals
    ##############################################
    def plot_IC_data_of_selected_pinns(self,N_batches,quarters_max_indices):
        global epoch
        # get for each pinn initial data (IC)
        num_x_intervals = int(np.sqrt(N_batches))
        num_y_intervals = int(np.sqrt(N_batches))
        step_x = max(self.x) / num_x_intervals
        step_y = max(self.y) / num_x_intervals  
        t_min=self.t_min
        t_max=self.t_max
        
        for phase_idx in range(self.num_phases): #this is a plot loop, no computng associated with 
            IC_nbr=0
            fig, ax = plt.subplots()       
            scatter_IC = plt.scatter([], [], cmap=plt.get_cmap('jet'))              
            for counter, pinn in enumerate(self.pinns):
                batch_idx = counter // self.num_phases
                pinn_id = counter % self.num_phases
                i = batch_idx  // num_x_intervals
                j = batch_idx %  num_x_intervals
                x_min = i * step_x
                x_max = (i + 1) * step_x
                y_min = j * step_y
                y_max = (j + 1) * step_y

                if  pinn_id==phase_idx and pinn.idx_pinn in quarters_max_indices:
                    phi=pinn.batch_phi_ini
                    IC_nbr+=len(phi)
                    X_Y=pinn.batch_X_ini
                    x_avg = (x_min + x_max) / 2
                    L=abs(x_max-x_min)
                    W= abs(y_max-y_min)
                    y_avg = (y_min + y_max) / 2
                    color = np.random.rand(3)
                    ax.text(x_avg, y_avg, f"{pinn.idx_pinn}", color=color, ha='right', va='bottom',fontsize=10)
                    ax.text(x_avg-L/4, y_avg-W/4, f"{len(phi)} pts", color='orange', ha='right', va='bottom',fontsize=8)
                    ax.scatter(X_Y[:, 0], X_Y[:, 1], color=color, marker='o',s=25, label='IC')
                    scatter_IC=ax.scatter(X_Y[:, 0], X_Y[:, 1], cmap=plt.get_cmap('viridis'), c=phi,marker='o',s=10,vmin=0,vmax=1)
       
            cbar = plt.colorbar(scatter_IC, ax=ax, shrink=0.35, label=r"$\phi$")
            
            title=f"IC_Reduced_points_{IC_nbr}_Epoch_{epoch}_Time_interval_t_min_{pinn.t_min:.5f}_t_max_{pinn.t_max:.5f}_Phase_{phase_idx}_N_batches_{N_batches}.jpg"
            #if epoch==0 or flag==1 or flag_reduce_batches==1 and flag_shuffle_for_scipy==0:
            plt.grid(True)
            plt.xticks(np.linspace(self.x.min(), self.x.max(), num_x_intervals+1))
            plt.yticks(np.linspace(self.y.min(), self.y.max(), num_y_intervals+1))
            plt.savefig(os.path.join("save_figs",title), dpi = 500, bbox_inches='tight') 
            
            plt.close()
    #############################################
    def plot_Collocation_data_of_selected_pinns(self, N_batches, quarters_max_indices, Nf):
        global epoch
        num_x_intervals = int(np.sqrt(N_batches))
        num_y_intervals = int(np.sqrt(N_batches))
        step_x = max(self.x) / num_x_intervals
        step_y = max(self.y) / num_y_intervals
        t_min = self.t_min
        t_max = self.t_max
        Col_nbr = 0
    
        for phase_idx in range(self.num_phases):
            fig, ax = plt.subplots()
            scatter = plt.scatter([], [], cmap=plt.get_cmap('jet'))
    
            for counter, pinn in enumerate(self.pinns):
                batch_idx = counter // self.num_phases
                pinn_id = counter % self.num_phases
    
                i = batch_idx // num_x_intervals
                j = batch_idx % num_x_intervals
                x_min = i * step_x
                x_max = (i + 1) * step_x
                y_min = j * step_y
                y_max = (j + 1) * step_y
    
                if pinn_id == phase_idx and pinn.idx_pinn in quarters_max_indices:
                    Col_nbr += len(pinn.batch_Xf)
                    X_Y_T = pinn.batch_Xf
    
                    x_avg = (x_min + x_max) / 2
                    L = abs(x_max - x_min)
                    W = abs(y_max - y_min)
                    y_avg = (y_min + y_max) / 2
                    color = np.random.rand(3)
    
                    scatter = plt.scatter(
                        X_Y_T[:, 0],
                        X_Y_T[:, 1],
                        cmap=plt.get_cmap('jet'),
                        c=X_Y_T[:, 2],
                        s=0.1
                    )
    
                    ax.text(
                        x_avg, y_avg,
                        f"{pinn.idx_pinn}",
                        color=color,
                        ha='right',
                        va='bottom',
                        fontsize=10
                    )
    
                    ax.text(
                        x_avg - L/4, y_avg - W/4,
                        f"{len(pinn.batch_Xf)} pts",
                        color='orange',
                        ha='right',
                        va='bottom',
                        fontsize=8
                    )
    
            cbar = plt.colorbar(scatter, shrink=0.5)
            cbar.set_label("Time")
    
            title = f"Collocation_Reduced_points_Epoch_{epoch}_Time_interval_t_min_{t_min:.5f}_t_max_{t_max:.5f}_Phase_{phase_idx}_N_batches_{N_batches}.jpg"
            plt.grid(True)
            plt.xticks(np.linspace(self.x.min(), self.x.max(), num_x_intervals+1))
            plt.yticks(np.linspace(self.y.min(), self.y.max(), num_y_intervals+1))
            plt.savefig(os.path.join("save_figs", title), dpi=500, bbox_inches='tight')
    
            plt.close()
  
    #############################################
    def set_Collocation_points_for_pinns(self,N_batches,Nf,flag_plot=True):
        num_x_intervals = int(np.sqrt(N_batches))
        num_y_intervals = int(np.sqrt(N_batches))
        step_x = max(self.x) / num_x_intervals
        step_y = max(self.y) / num_x_intervals  
        t_min=self.t_min
        t_max=self.t_max
        
        Col_nbr=0
        for phase_idx in range(self.num_phases): #this is a plot loop, no computng associated with 
            fig, ax = plt.subplots()                    
            for counter, pinn in enumerate(self.pinns):
                
                batch_idx = counter // self.num_phases
                pinn_id = counter % self.num_phases
                
                n=len(self.X_ini_all_sub_domain)
                N=self.num_phases
                i = batch_idx  // num_x_intervals
                j = batch_idx %  num_x_intervals
                x_min = i * step_x
                x_max = (i + 1) * step_x
                y_min = j * step_y
                y_max = (j + 1) * step_y
                
                batch_X_ini=pinn.batch_X_ini
                batch_phi_ini=pinn.batch_phi_ini
                
                global epoch 
                batch_Xf=self.generate_collocation_points(batch_X_ini,batch_phi_ini, Nf, x_min, x_max, y_min, y_max,epoch)
                pinn.batch_Xf=batch_Xf
                
                if  pinn_id==phase_idx:
                    Col_nbr+= len(batch_Xf)
                    X_Y_T=batch_Xf
                    # DEBUG
                    x_avg = (x_min + x_max) / 2
                    L=abs(x_max-x_min)
                    W= abs(y_max-y_min)
                    y_avg = (y_min + y_max) / 2
                    color = np.random.rand(3)
                    plt.scatter(X_Y_T[:, 0], X_Y_T[:, 1], cmap=plt.get_cmap('jet'),c=X_Y_T[:, 2], s=0.1)
                    ax.text(x_avg, y_avg, f"{pinn.idx_pinn}", color=color, ha='right', va='bottom',fontsize=10)
                    ax.text(x_avg-L/4, y_avg-W/4, f"{len(batch_Xf)} pts", color='orange', ha='right', va='bottom',fontsize=8)
      
            #cbar =plt.colorbar( shrink=0.5)                              
            #cbar.set_label("Time")

            title=f"Collocation_points_{Col_nbr}_Epoch_{epoch}_for_Time_interval_t_min_{pinn.t_min:.5f}_t_max_{pinn.t_max:.5f}_Phase_{phase_idx}.jpg"
            plt.grid(True)
            plt.xticks(np.linspace(self.x.min(), self.x.max(), num_x_intervals+1))
            plt.yticks(np.linspace(self.y.min(), self.y.max(), num_y_intervals+1))
            if flag_plot==True:
              plt.savefig(os.path.join("save_figs",title), dpi = 500, bbox_inches='tight') 

            plt.close()        
    ##############################################
    def get_and_plot_reduced_IC_subdomains(self,epoch,N_batches,N_ini_min_per_batch,N_ini_max_per_batch,\
        fraction_ones_per_int_pts,fraction_zeros_per_int_pts,path,flag_multi_phases=False):
        X_lb_sub_domain=self.X_lb_sub_domain
        X_ltb_sub_domain=self.X_ltb_sub_domain
        all_indices_sampled=self.get_all_indices_sampled(self,fraction_ones_per_int_pts,fraction_zeros_per_int_pts)
        selected_indices_scipy = []  # IC indices for scipy optimizer
        num_x_intervals = int(np.ceil(np.sqrt(N_batches)))
        num_y_intervals = int(np.ceil(np.sqrt(N_batches)))
        x_interval_size = (X_lb_sub_domain[:, 0].max() - X_lb_sub_domain[:, 0].min()) / num_x_intervals
        y_interval_size = (X_ltb_sub_domain[:, 1].max() - X_ltb_sub_domain[:, 1].min()) / num_y_intervals
        
        Percentages_interface_points=[]
        Total_list_indices_ini=[]
        X_ini_all_sub_domain_reduced=self.X_ini_all_sub_domain[all_indices_sampled] # containing interfacial points (n %) and some grain points (10%)
        phi_ini_all_sub_domain_reduced=self.phi_ini_all_sub_domain[all_indices_sampled]
    
        sublimits_list = []
        step_x = max(self.x) / num_x_intervals
        step_y = max(self.y) / num_x_intervals
        
        for i in range(num_x_intervals):
            for j in range(num_y_intervals):
                
                # Define the x and y bounds for the current interval
                x_lb = X_lb_sub_domain[:, 0].min() + i * x_interval_size
                x_ub = X_lb_sub_domain[:, 0].min() + (i + 1) * x_interval_size
                y_lb = X_ltb_sub_domain[:, 1].min() + j * y_interval_size
                y_ub = X_ltb_sub_domain[:, 1].min() + (j + 1) * y_interval_size

                # Numerotation of pinns 
                x_min = i * step_x
                x_max = (i + 1) * step_x
                y_min = j * step_y
                y_max = (j + 1) * step_y
                sublimits_list.append([x_min, x_max, y_min, y_max])

                # Find the indices of points within the current interval
                batch_indices = np.where((X_ini_all_sub_domain_reduced[:, 0] >= x_lb) & (X_ini_all_sub_domain_reduced[:, 0] <= x_ub) &
                                            (X_ini_all_sub_domain_reduced[:, 1] >= y_lb) & (X_ini_all_sub_domain_reduced[:, 1] <= y_ub))[0]
                                            
                sum_phi_columns = np.sum(self.phi_ini_all_sub_domain[all_indices_sampled], axis=1)
                
                int_phi_values=sum_phi_columns[batch_indices]

                int_thresh=self.thresh_interface # a small filter to not to consider noisy areas 
                total_interface_points = np.sum((int_phi_values > int_thresh) & (int_phi_values < 1 - int_thresh))
                
                #tf.print("total_interface_points: ",i * num_y_intervals + j,total_interface_points,len(batch_indices) )
                if len(batch_indices)>0:
                    percentage_interface_points = total_interface_points / len(np.asarray(self.phi_ini_all_sub_domain[all_indices_sampled])[batch_indices])
                else:
                    percentage_interface_points=0
                                            
                Percentages_interface_points.append(percentage_interface_points)
                #tf.print("total_interface_points: ", i * num_y_intervals + j,percentage_interface_points,N_ini_max_per_batch,N_ini_max_per_batch)
                N_ini_per_batch = max(int(percentage_interface_points *N_ini_min_per_batch), N_ini_min_per_batch)

                if len(batch_indices) > 0:   # number of points per batch
                    #tf.print("size: ",len(batch_indices),max(N_ini_per_batch,N_ini_min_per_batch))
                    if len(batch_indices)<max(N_ini_per_batch,N_ini_min_per_batch):
                        replace=True  
                    else:
                        replace =False 

                    list_ini_per_batch=np.random.choice(batch_indices, size=min(N_ini_per_batch,N_ini_max_per_batch), replace=False)
                    selected_indices_scipy.extend(list_ini_per_batch)
                    Total_list_indices_ini.append(list_ini_per_batch)
        self.indices_ini =selected_indices_scipy # update indices for scipy optimization
                                                # for each subdomain self.indices will return the IC points  
        
        X_ini_all_sub_domain_reduced=np.asarray(X_ini_all_sub_domain_reduced)
        phi_ini_all_sub_domain_reduced=np.asarray(phi_ini_all_sub_domain_reduced)
        
        X_ini_sub_domain = X_ini_all_sub_domain_reduced[selected_indices_scipy]
        phi_ini_sub_domain = phi_ini_all_sub_domain_reduced[selected_indices_scipy] 

        self.X_ini_sub_domain=X_ini_all_sub_domain_reduced[self.indices_ini]
        self.phi_ini_sub_domain=phi_ini_all_sub_domain_reduced[self.indices_ini]                    
        fig, ax = plt.subplots()
        # numerotaion of pinns 
        for i in range(N_batches):
            x_min, x_max, y_min, y_max=sublimits_list[i]
            x_avg = (x_min + x_max) / 2
            y_avg = (y_min + y_max) / 2
            plt.text(x_avg, y_avg, f"{i}", color='black', ha='right', va='bottom',fontsize=5)

        plt.scatter(X_ini_sub_domain[:, 0], X_ini_sub_domain[:, 1],s=0.5,\
            cmap=plt.get_cmap('jet'), c=np.sum(phi_ini_sub_domain,axis=1))
        plt.colorbar( shrink=0.35)
        t_min=self.t_min
        t_max=self.t_max
        title=f"Global IC points at Epoch {epoch} for Time interval: t_min: {t_min:.5f}, t_max: {t_max:.5f}.jpg"
        phi_ini_length = len(np.asarray(self.phi_ini_sub_domain))
        plt.title(f'Number of IC points for Scipy and ADAM optimization: {phi_ini_length}',fontsize=8)
        plt.grid(True)
        plt.xticks(np.linspace(self.lb[0], self.ub[0], num_x_intervals+1))
        plt.yticks(np.linspace(self.lb[1], self.ub[1], num_y_intervals+1))
        if epoch==0 or flag==1 or flag_reduce_batches==1 and flag_shuffle_for_scipy==0:
            plt.savefig(os.path.join(path,title), dpi = 500, bbox_inches='tight')
        plt.close()  
    ############################################### 
    def update_batch_Xf(self):
        batch_Xf_dict = {}

        def compute_batch_Xf_for_batch(batch_pinns):
            combined_batch_Xf = np.concatenate([pinn.batch_Xf for pinn in batch_pinns], axis=0)
            return combined_batch_Xf
        
        # Update batch_Xf for all pinns
        for pinn_idx_alpha, pinn_alpha in enumerate(self.pinns):
            idx_batch_alpha = pinn_alpha.idx_batch
            batch_Xf_alpha = pinn_alpha.batch_Xf  
        
            # Check if batch_Xf for this batch has already been computed
            if idx_batch_alpha in batch_Xf_dict:
                # If it has, use the existing batch_Xf 
                pinn_alpha.batch_Xf = batch_Xf_dict[idx_batch_alpha]
            else:
                batch_pinns_in_same_batch = [pinn for pinn in self.pinns if pinn.idx_batch == idx_batch_alpha]
                batch_Xf_alpha = compute_batch_Xf_for_batch(batch_pinns_in_same_batch)
                batch_Xf_dict[idx_batch_alpha] = batch_Xf_alpha
                pinn_alpha.batch_Xf = batch_Xf_alpha
        
            # using batch_Xf_dict, update batch_Xf for other pinns in the same batch
            for pinn_idx_beta, pinn_beta in enumerate(self.pinns):
                idx_batch_beta = pinn_beta.idx_batch
                if (pinn_idx_alpha != pinn_idx_beta) and (idx_batch_alpha == idx_batch_beta):
                    pinn_beta.batch_Xf = pinn_alpha.batch_Xf

        #DEBUG
        """
        os.makedirs("check_batches", exist_ok=True)
        for pinn_idx_alpha, pinn_alpha in enumerate(self.pinns):
            idx_batch_alpha = pinn_alpha.idx_batch
            batch_Xf_alpha = pinn_alpha.batch_Xf  
            
            for pinn_idx_beta, pinn_beta in enumerate(self.pinns):
                idx_batch_beta = pinn_beta.idx_batch
                batch_Xf_beta = pinn_beta.batch_Xf  
                
                if (pinn_idx_alpha != pinn_idx_beta) and (idx_batch_alpha == idx_batch_beta):
                    if not np.array_equal(batch_Xf_alpha, batch_Xf_beta):
                        
                        tf.print(f"PINN {pinn_alpha.idx_pinn} and PINN {pinn_beta.idx_pinn} in the same batch {idx_batch_alpha} have different batch_Xf.")
                    
                    fig, ax = plt.subplots()
                    tf.print(f"PINN {pinn_alpha.idx_pinn} and PINN {pinn_beta.idx_pinn} in the same batch {idx_batch_alpha} have same batch_Xf.")                                    
                    #tf.print(batch_Xf_alpha)
                    plt.scatter(batch_Xf_alpha[:, 0], batch_Xf_alpha[:, 1], cmap=plt.get_cmap('viridis'), c=batch_Xf_alpha[:, 2])
                    title = f"batch_Xf_alpha_{pinn_alpha.idx_pinn}_len: {len(pinn_alpha.batch_Xf)}.jpg"
                    plt.savefig(os.path.join("check_batches", title))
                    plt.close()
        """
        for pinn in self.pinns:
            #tf.print(int(pinn.idx_phase))
            if int(pinn.idx_phase) != 0:
                del pinn.batch_Xf        
                #tf.print(f"deleted batch_Xf for pinn{pinn.idx_pinn}")
 
    ###############################################
    def update_batch_X_ini(self):
        # Define the percentage of coordinates to append (10%)
        append_percentage = 0.1
        """
        # Update batch_Xf for all pinns
        for pinn_idx_alpha, pinn_alpha in enumerate(self.pinns):
            idx_batch_alpha = pinn_alpha.idx_batch
            batch_X_ini_alpha = pinn_alpha.batch_X_ini 
            batch_phi_ini_alpha = pinn_alpha.batch_phi_ini 
            
            batch_pinns_in_same_batch = [pinn for pinn in self.pinns if pinn.idx_batch == idx_batch_alpha]

            # using batch_pinns_in_same_batch, update batch_phi_ini_alpha by some coordinates with 0 phi values
            for pinn_idx_beta, pinn_beta in enumerate(batch_pinns_in_same_batch):
                idx_batch_beta = pinn_beta.idx_batch
                if (pinn_idx_alpha != pinn_idx_beta) and (idx_batch_alpha == idx_batch_beta):
                    beta_batch_X_ini=pinn_beta.batch_X_ini 
                    num_samples_to_append = int(len(beta_batch_X_ini) * append_percentage)
                    zero_indices = np.random.choice(len(beta_batch_X_ini), size=num_samples_to_append, replace=False)
                    append_batch_X_ini=beta_batch_X_ini[zero_indices]
                    append_batch_phi_ini = np.zeros((len(append_batch_X_ini), 1)) 
                    
                    pinn_alpha.batch_X_ini  = np.concatenate([pinn_alpha.batch_X_ini ,append_batch_X_ini], axis=0)
                    pinn_alpha.batch_phi_ini  = np.concatenate([pinn_alpha.batch_phi_ini ,append_batch_phi_ini], axis=0)
        """
        os.makedirs("check_batches", exist_ok=True)
        for pinn_idx_alpha, pinn_alpha in enumerate(self.pinns):
            idx_batch_alpha = pinn_alpha.idx_batch
            batch_X_ini_alpha = pinn_alpha.batch_X_ini  
            batch_phi_ini_alpha = pinn_alpha.batch_phi_ini 
            
            for pinn_idx_beta, pinn_beta in enumerate(self.pinns):
                idx_batch_beta = pinn_beta.idx_batch
                batch_X_ini_beta = pinn_beta.batch_X_ini  
                batch_X_ini_beta = pinn_beta.batch_phi_ini 
                
                if (pinn_idx_alpha != pinn_idx_beta) and (idx_batch_alpha == idx_batch_beta):
                    fig, ax = plt.subplots()

                    scatter_IC=plt.scatter(batch_X_ini_alpha[:, 0], batch_X_ini_alpha[:, 1], cmap=plt.get_cmap('viridis'), c=batch_phi_ini_alpha, vmin=0, vmax=1)
                    cbar = plt.colorbar(scatter_IC, ax=ax, shrink=0.35, label=r"$\phi$")
                    title = f"batch_X_ini_alpha_{pinn_alpha.idx_pinn}_len: {len(pinn_alpha.batch_X_ini)}.jpg"
                    plt.savefig(os.path.join("check_batches", title))
                    plt.close()
        
    ###############################################
    def get_data_next_time_interval(self,N_batches): 
        global path_save
        X_ini_all=[]
        phi_ini_all=[]

        fig, axs = plt.subplots(self.num_phases, 1, figsize=(10, 20)) 
        fig.subplots_adjust(hspace=0.8, wspace=0.4)
        num_x_intervals = int(np.sqrt(N_batches))
        num_y_intervals = int(np.sqrt(N_batches))
        step_x = max(self.x) / num_x_intervals
        step_y = max(self.y) / num_x_intervals  
        count=1
        for idx_phase in range(self.num_phases):
            X_phase=[]
            phi_phase=[]

            for counter, pinn in enumerate(self.pinns):
                batch_idx = counter // self.num_phases
                pinn_id = counter % self.num_phases
                n=len(self.X_ini_all_sub_domain)
                N=self.num_phases
                X_ini_all_sub_domain_phase= self.X_ini_all_sub_domain[int(n*pinn_id/N)+1:int(n*(pinn_id+1)/N)]
                i = batch_idx  // num_x_intervals
                j = batch_idx %  num_y_intervals
                x_min = i * step_x
                x_max = (i + 1) * step_x
                y_min = j * step_y
                y_max = (j + 1) * step_y
                X_ini_all_sub_domain_pinn=X_ini_all_sub_domain_phase[pinn.all_ini_indices]
                X_ini_all_sub_domain_pinn[:,2]=self.t_max
                phi_ini_all_sub_domain_pinn=pinn.evaluate(X_ini_all_sub_domain_pinn)
                
                if  int(pinn.idx_phase)==idx_phase:
                    X_phase.append(X_ini_all_sub_domain_pinn)
                    phi_phase.append(phi_ini_all_sub_domain_pinn)
            X_phase = np.vstack(X_phase)  
            phi_phase = np.vstack(phi_phase)  
                
            X_ini_all.append(X_phase)
            phi_ini_all.append(phi_phase)

            plt.scatter(X_phase[:, 0], X_phase[:, 1], cmap=plt.get_cmap('viridis'), c=phi_phase)
            plt.title(f"Phi of Phase {idx_phase}")
            
        dt=self.t_max-self.t_min
        save_path = os.path.join(path_save, f"X_ini_all_phi_ini_all_t_min_{self.t_max}_t_max_{self.t_max+dt}") 
        plt.savefig(save_path+".jpg")
        plt.close()
        
        X_ini_all = np.vstack(X_ini_all)  
        phi_ini_all = np.vstack(phi_ini_all)
        
        #elf.X_ini_all_sub_domain=X_ini_all 
        #self.phi_ini_all_sub_domain=phi_ini_all
        #tf.print("out", self.X_ini_all_sub_domain.shape, self.phi_ini_all_sub_domain.shape)
    ###############################################
    def set_average_weights_and_save(self, weights_key=0):
        t_min=self.t_min
        t_max=self.t_max
        
        weights_list = []
        for pinn in self.pinns:
            weights_list.append(pinn.get_weights())
        # compute the average of the weights and biases
        average_weights = weights_list[0]#sum(weights_list) / len(weights_list)
        #tf.print("Heeere:",np.asarray(average_weights).shape, len(weights_list) )
        self.set_weights(average_weights)
        weights_file = 'weights/weights_tmin_{:.5f}_tmax_{:.5f}_{}.json'.format(t_min, t_max, weights_key)
        self.save_weights(self,weights_file)    
    ###############################################        
    def save_weights_for_pinns(self, N_batches, path_weights_all_pinns, t_min, t_max,quarters_max_indices, weights_key=0):
        if t_min==0: 
            self.PRE_POST.EraseFile(path=path_weights_all_pinns)
        
        for pinn in self.pinns:
            pinn_idx_batch = pinn.idx_batch
            pinn_idx_pinn = pinn.idx_pinn
            
            # folder (pinn.idx_batch)
            parent_folder = os.path.join(path_weights_all_pinns, f"batch_{pinn_idx_batch}")
            if not os.path.exists(parent_folder):
                os.makedirs(parent_folder)
            
            #  folder (pinn.idx_pinn) within the parent folder
            child_folder = os.path.join(parent_folder, f"pinn_{pinn_idx_pinn}")
            if not os.path.exists(child_folder):
                os.makedirs(child_folder)
        
        tf.print("\n !!!  saving weights of self.pinns in progress !!! \n")
        num_processes = min(multiprocessing.cpu_count(), N_batches)  
        pool = multiprocessing.pool.ThreadPool(processes=num_processes)
        all_pinns_args_weights = []
        for pinn in self.pinns:
            if pinn.idx_pinn in quarters_max_indices:
                parent_folder = os.path.join(path_weights_all_pinns, f"batch_{pinn.idx_batch}")
                pinn_folder = os.path.join(parent_folder, f"pinn_{pinn.idx_pinn}")
                pinn_path_weights = os.path.join(pinn_folder, f"pinn_{pinn.idx_pinn}_weights_tmin_{t_min:.5f}_tmax_{t_max:.5f}_{weights_key}.json")
                #tf.print(pinn.idx_pinn, pinn_path_weights)
                all_pinns_args_weights.append([pinn, pinn_path_weights])
                self.save_weights(pinn,pinn_path_weights)
        #pool.starmap(self.save_weights, all_pinns_args_weights)
        #pool.close()
        #f.join()
        tf.print("\n     ! saving complete ! \n")
    ###############################################
    def set_weights_all_pinns(self,N_batches,path_in):
        tf.print("\n ! set initial weights for self.pinns! \n")
        path_get_weights_all_pinns=path_in
        weights_directories_in = [f for f in os.listdir(path_get_weights_all_pinns) if os.path.isdir(os.path.join(path_get_weights_all_pinns, f))]
        N_batches_small = len(weights_directories_in)
        # Check if the number of subdirectories matches N_batches
        if N_batches_small == N_batches:
            for pinn_idx in range(len(self.pinns)):
                pinn = self.pinns[pinn_idx]
                pinn_path_weights = os.path.join(path_get_weights_all_pinns, f"batch_{pinn.idx_batch}/pinn_{pinn.idx_pinn}")
                weights_files = glob.glob(os.path.join(pinn_path_weights, '*.json'))
                weights_files = sorted(weights_files)
                
                if not weights_files:
                    #tf.print(f"No weight files found for pinn {pinn.idx_pinn} in batch {pinn.idx_batch}.")
                    continue
                
                if weights_files:
                    weights_file = weights_files[-1]
                    #tf.print(weights_file)

                    with open(weights_file, 'r') as f:
                        weights_loaded = json.load(f)['weights']
                    weights_loaded = tf.cast(weights_loaded, dtype=self.precision)
                    pinn.set_weights(weights_loaded)


                with open(weights_file, 'r') as f:
                    weights_loaded =json.load(f)['weights']
                weights_loaded=tf.cast(weights_loaded, dtype=self.precision)  
                pinn.set_weights(weights_loaded) 
                pinn.loss_value=1
                
        else:  # suppose to transfer the learning from 4  to 4* N
            matrix_size = int(np.sqrt(N_batches))
            matrix = np.arange(N_batches).reshape(matrix_size, matrix_size)
            quarter1 = matrix[:matrix_size//2, :matrix_size//2].flatten()
            quarter2 = matrix[:matrix_size//2, matrix_size//2:].flatten()
            quarter3 = matrix[matrix_size//2:, :matrix_size//2].flatten()
            quarter4 = matrix[matrix_size//2:, matrix_size//2:].flatten()
            quarters = {
                    'quarter1': quarter1,
                    'quarter2': quarter2,
                    'quarter3': quarter3,
                    'quarter4': quarter4
                }
            Quarters=[quarter1,quarter2,quarter3,quarter4] 
            tf.print("\n    ! Multiple to Multiple transfer learning ! \n")

            for i, weights_dir in enumerate(weights_directories_in):
                weights_path = os.path.join(path_get_weights_all_pinns, weights_dir)
                weights_files = glob.glob(os.path.join(weights_path, '*.json'))
                weights_files = sorted(weights_files)
                quarter=Quarters[i]
                for j in quarter: 
                    #tf.print("here, i,j: ", i,j)
                    pinn = self.pinns[j]
                    pinn_path_weights =os.path.join(os.getcwd(),os.path.join(path_get_weights_all_pinns, f"pinn_{i}") )
                    #tf.print(pinn_path_weights)
                    weights_files = glob.glob(os.path.join(pinn_path_weights, '*.json'))
                    weights_files = sorted(weights_files)
                    weights_file = weights_files[-1]
                    match = re.search(r'_tmin_(\d+\.\d+)_tmax_(\d+\.\d+)_', weights_file)

                    if match:
                        t_min = float(match.group(1))
                        t_max = float(match.group(2))
                    else:
                        print("Unable to extract t_min and t_max.")

                    with open(weights_file, 'r') as f:
                        weights_loaded = json.load(f)['weights']
                    weights_loaded = tf.cast(weights_loaded, dtype=self.precision)
                    pinn.set_weights(weights_loaded)
            tf.print("\n    ! set weights completed ! \n") 
    ###############################################
    def set_new_limits_Master(self ): 
        N_batches= self.N_batches
        limits = []
        num_x_intervals = int(np.sqrt(N_batches))
        num_y_intervals = int(np.sqrt(N_batches))
        step_x = max(self.x) / num_x_intervals
        step_y = max(self.y) / num_y_intervals
        for i in range(int(np.sqrt(N_batches))):
            for j in range(int(np.sqrt(N_batches))):
                #tf.print("index", index)
                x_min = i * step_x
                x_max = (i + 1) * step_x
                y_min = j * step_y
                y_max = (j + 1) * step_y
                limits.append([x_min, x_max, y_min, y_max])
        
        for reset_pinn_idx in sorted(range(len(self.pinns) )):
            self.pinns[reset_pinn_idx].limits=limits[reset_pinn_idx]
            self.pinns[reset_pinn_idx].idx_batch=reset_pinn_idx        
    ###############################################
    def set_pinns_limits(self ): 
        N_batches= len(self.pinns)  
        limits = []
        num_x_intervals = int(np.sqrt(N_batches))
        num_y_intervals = int(np.sqrt(N_batches))
        step_x = max(self.x) / num_x_intervals
        step_y = max(self.y) / num_y_intervals
        for i in range(int(np.sqrt(N_batches))):
            for j in range(int(np.sqrt(N_batches))):
                #tf.print("index", index)
                x_min = i * step_x
                x_max = (i + 1) * step_x
                y_min = j * step_y
                y_max = (j + 1) * step_y
                limits.append([x_min, x_max, y_min, y_max]) 
        for pinn_idx in range(len(self.pinns) ):
            self.pinns[pinn_idx].limits=limits[pinn_idx]

    ###############################################   
    def set_new_limits(self ): 
        N_batches= len(self.pinns)  
        limits = []
        num_x_intervals = int(np.sqrt(N_batches))
        num_y_intervals = int(np.sqrt(N_batches))
        step_x = max(self.x) / num_x_intervals
        step_y = max(self.y) / num_y_intervals
        for i in range(int(np.sqrt(N_batches))):
            for j in range(int(np.sqrt(N_batches))):
                #tf.print("index", index)
                x_min = i * step_x
                x_max = (i + 1) * step_x
                y_min = j * step_y
                y_max = (j + 1) * step_y
                limits.append([x_min, x_max, y_min, y_max])
        
        for reset_pinn_idx in sorted(range(len(self.pinns) )):
            self.pinns[reset_pinn_idx].limits=limits[reset_pinn_idx]
            self.pinns[reset_pinn_idx].idx_batch=reset_pinn_idx

    ###############################################
    def is_intersection(self,square1, square2):
        x_min1, x_max1,y_min1, y_max1 = square1
        x_min2, x_max2, y_min2, y_max2 = square2
        x_avg_1 = (x_min1 + x_max1) / 2
        y_avg_1 = (y_min1 + y_max1) / 2
        x_avg_2 = (x_min2 + x_max2) / 2
        y_avg_2 = (y_min2 + y_max2) / 2
        dist_centers=np.sqrt((x_avg_1-x_avg_2)**2 +(y_avg_1-y_avg_2)**2)
        a_1=np.abs((x_min1- x_max1))
        b_1=np.abs((y_min1- y_max1))
        a_2=np.abs((x_min2- x_max2))
        b_2=np.abs((y_min2-y_max2))
        r1=a_1/2
        r2=a_2/2
        return  (dist_centers < (r1+r2) )
    ###############################################
    def check_intersection(self,pinn, domain_limit):
        return self.is_intersection(pinn.limits, domain_limit)
            
    ###############################################
    def reduce_batches_Pyramid_like_training(self,N_batches,num_x_intervals,num_y_intervals,\
    path_weights_all_pinns,\
            coef_increase_points_f,coef_increase_points_ic,\
                Nbr_f_pts_max_per_batch,Nbr_f_pts_min_per_batch,\
                    N_ini_max_per_batch, N_ini_min_per_batch,quarters_max_indices,dict_quarters_infos  ):
        old_pinns=np.copy(self.pinns)
 
        ############################## 
        def set_pinns_limits():
            #sublimits_next_list
            offset=int(np.sqrt(N_batches)) 
            sublimits_next_list = []
            step_x = max(self.x) / np.sqrt(N_batches)
            step_y = max(self.y) / np.sqrt(N_batches)         
            for i in range(int(np.sqrt(N_batches))):
                for j in range(int(np.sqrt(N_batches))):
                    x_min = i * step_x
                    x_max = (i + 1) * step_x
                    y_min = j * step_y
                    y_max = (j + 1) * step_y
   
        ############################## 
        
         
        matrix_size = int(np.sqrt(N_batches))
        matrix = np.arange(N_batches).reshape(matrix_size, matrix_size)
        quarter1 = matrix[:matrix_size//2, :matrix_size//2].flatten()
        quarter2 = matrix[:matrix_size//2, matrix_size//2:].flatten()
        quarter3 = matrix[matrix_size//2:, :matrix_size//2].flatten()
        quarter4 = matrix[matrix_size//2:, matrix_size//2:].flatten()
        Quarters=[quarter1,quarter2,quarter3,quarter4]                        
        #tf.print("\n quarter1:", ", ".join(map(str, quarter1)))
        #tf.print("quarter2:", ", ".join(map(str, quarter2)))
        #tf.print("quarter3:", ", ".join(map(str, quarter3)))
        #tf.print("quarter4:", ", ".join(map(str, quarter4)))
                  
        quarters = [quarter1, quarter2, quarter3, quarter4]
        
        ##############################  
        def extract_batch_phase(idx_pinn):
            batch = int(str(idx_pinn)[:2])
            phase = int(str(idx_pinn)[2:])
            return batch, phase
        ##############################
        def return_the_idx_pinn_which_is_in_dict_quarters_infos(phase, quarter, dict_quarters_infos):  
            # Find the corresponding idx_pinn value in dict_quarters_infos
            key = f'quarter{quarter}'
            if key in dict_quarters_infos[f'Phase_{phase}']:
               return dict_quarters_infos[f'Phase_{phase}'][key][0]['idx_pinn']
            return None
        ##############################  
        def initialize_new_pinns(N_batches):               
            pinns = []  # List to store the PINN objects
            for batch_idx in range(N_batches):
                for pinn_id in range(self.num_phases):
                    pinn = Sequentialmodel(layers=self.layers, X_f_train=self.X_f, X_ini_train=self.X_ini,\
                                            phases_ini_indexes=self.phases_ini_indexes,all_ini_flags_matrix=self.All_flag_ini,\
                                            Phi_ini=self.All_phi_ini,phi_ini_train=self.phi_ini, N_ini=self.N_ini,X_phi_test=self.X_phi_test,\
                                            X_ini_train_all=self.X_ini_train_all, phi_ini_train_all=self.phi_ini_train_all,\
                                                all_interfaces=self.All_interfaces_ini,\
                                            X_lb_train=self.X_lb, X_ub_train=self.X_rtb,\
                                            X_ltb_train=self.X_ltb, X_rtb_train=self.X_rtb,\
                                            X=None,Y=None,T=None,x=self.x,y=self.y,lb=self.lb, ub=self.ub, mu=self.mu, sigma=self.sigma, delta_g=self.delta_g,\
                                                eta=self.eta,Nx=self.Nx,Ny=self.Ny,Nt=self.Nt,phi_sol=None,pinns =self.pinns,num_phases=self.num_phases,
                                            N_batches=N_batches,
                                            min_batch_numbers = self.min_batch_numbers,\
                                            Nbr_f_pts_max_per_batch=self.Nbr_f_pts_max_per_batch,\
                                            Nbr_f_pts_min_per_batch=self.Nbr_f_pts_min_per_batch,\
                                            N_ini_max_per_batch=self.N_ini_max_per_batch,\
                                            N_ini_min_per_batch=self.N_ini_min_per_batch)   
                                    

                    pinn.idx_batch = batch_idx 
                    pinn.idx_pinn = str(batch_idx).zfill(2)  + str(pinn_id).zfill(2) # if two-digit representation
                    pinn.idx_phase =str(pinn_id).zfill(2)
                    pinns.append(pinn) 
            return pinns
          
        new_pinns= initialize_new_pinns(N_batches)              

        ##############################  
        def look_for_batch_in_Quarters(batch, Quarters):
            # Iterate through quarters and find the quarter containing the batch
            for quarter, quarter_batches in enumerate(Quarters, start=1):
                if batch in quarter_batches:
                    return quarter
            return None            
        ##############################    
        def check_mapping(new_pinns, dict_quarters_infos):
            for pinn in new_pinns:
                # Extract batch, phase, and quarter information from the current PINN
                batch = int(pinn.idx_batch)
                phase = int(pinn.idx_phase)
        
                # Find the quarter for the current PINN based on its batch index
                quarter_new_pinn = look_for_batch_in_Quarters(batch,Quarters)
                
                # we have the phase and th quarter, we can identifiy the corresponind pinn
                # Find the corresponding idx_pinn value from quarters_max_indices
                idx_pinn= return_the_idx_pinn_which_is_in_dict_quarters_infos(phase,quarter_new_pinn,dict_quarters_infos)
                corresponding_pinn=self.find_pinn_by_idx_pinn(idx_pinn, old_pinns)
                pinn.set_weights(corresponding_pinn.get_weights())
                tf.print(f'New subdivision: N_batches {N_batches}, Phase {phase}, Batch {batch}, Quarter {quarter_new_pinn}, New PINN {pinn.idx_pinn}, Corresponding PINN from quarters_max_indices {idx_pinn}')
        ##############################  

        # Call the function to check the mapping
        check_mapping(new_pinns, dict_quarters_infos)

        ################################
        ######### Wichtig ##############
        ################################
        # here a new list of  pinns is selected
        # for next training 
        self.pinns=new_pinns

 

        # update sizes training batches 
        Nbr_f_pts_max_per_batch=int(Nbr_f_pts_max_per_batch *coef_increase_points_f)
        Nbr_f_pts_min_per_batch= int(Nbr_f_pts_min_per_batch*coef_increase_points_f)
        N_ini_max_per_batch=int(N_ini_max_per_batch*coef_increase_points_ic)
        N_ini_min_per_batch=int(N_ini_min_per_batch*coef_increase_points_ic )     
        #tf.print("Nbr_f_pts_max_per_batch:", Nbr_f_pts_max_per_batch)
        #tf.print("Nbr_f_pts_min_per_batch:", Nbr_f_pts_min_per_batch)
        #tf.print("N_ini_max_per_batch:", N_ini_max_per_batch)
        #tf.print("N_ini_min_per_batch:", N_ini_min_per_batch)
        
        del old_pinns
 
        return  Nbr_f_pts_max_per_batch,Nbr_f_pts_min_per_batch,N_ini_max_per_batch,N_ini_min_per_batch
    ############################################### 
    def generate_collocation_points_2(self,batch_X_ini,batch_phi_ini, Nf, x_min, x_max, y_min, y_max,epoch, idx_batch=0,quarters_max_indices=[]):
        a_square=abs(x_max-x_min)
        batch_X_f = []
        for index, point in enumerate(batch_X_ini):
            
            x, y, t_min = point

            phi = batch_phi_ini[index] 
            
            if phi<=self.thresh or phi>=1-self.thresh:
                coef_reduction=self.coef_reduction
                number_points_per_batch=int(Nf)
            else:
                coef_reduction=self.coef_reduction
                number_points_per_batch=Nf
                
            for _ in range(number_points_per_batch):
                theta = random.uniform(0, 2 * math.pi)
                radius =  2*self.eta* math.sqrt(random.uniform(0, 1))  #(a_square / coef_reduction)
                collocation_x = x + radius * math.cos(theta)
                collocation_y = y + radius * math.sin(theta)

                # Ensure generated points are within the spatial square
                collocation_x = max(min(collocation_x, x_max), x_min)
                collocation_y = max(min(collocation_y, y_max), y_min)
                collocation_t = random.uniform(self.t_min, self.t_max)
                batch_X_f.append([collocation_x, collocation_y, collocation_t])

        batch_X_f = np.array(batch_X_f)  
        return batch_X_f     
    ############################################### 
    def generate_collocation_points(self, batch_X_ini, batch_phi_ini, Nf, x_min, x_max, y_min, y_max, epoch, idx_batch=0, quarters_max_indices=[]):
        a_square = abs(x_max - x_min)
        batch_X_f = []
    
        for index, point in enumerate(batch_X_ini):
            x, y, t_min = point
            phi = batch_phi_ini[index]
    
            # Skip points that are outside the interface
            #if not (self.thresh <= phi <= 1 - self.thresh):
            #    continue
    

            number_points_per_batch = Nf
            distance_to_x_edge = min(x - x_min, x_max - x)
            distance_to_y_edge = min(y - y_min, y_max - y)
            max_radius = min(self.eta / 3, distance_to_x_edge, distance_to_y_edge)
            radii = np.random.uniform(0, max_radius, number_points_per_batch)
            thetas = np.random.uniform(0, 2 * np.pi, number_points_per_batch)
            collocation_x = x + radii * np.cos(thetas)
            collocation_y = y + radii * np.sin(thetas)
            collocation_x = np.clip(collocation_x, x_min, x_max)
            collocation_y = np.clip(collocation_y, y_min, y_max)
            collocation_t = np.random.uniform(self.t_min, self.t_max, number_points_per_batch)
            generated_points = np.column_stack((collocation_x, collocation_y, collocation_t))
            batch_X_f.extend(generated_points.tolist())
    
        batch_X_f = np.array(batch_X_f)
    
        # filter points in batch_X_f based on proximity to batch_X_ini
        distances = np.linalg.norm(batch_X_f[:, :2][:, np.newaxis] - batch_X_ini[:, :2], axis=2)
    
        #  minimum distance for each point in batch_X_f
        min_distances = np.min(distances, axis=1)
    
        # Filter batch_X_f based on proximity to batch_X_ini within self.eta/5
        filtered_batch_X_f = batch_X_f[min_distances < self.eta/5]
    
        return filtered_batch_X_f

    ############################################### 
    def increase_points(self, Nbr_f_pts_max_per_batch, Nbr_f_pts_min_per_batch):
        factor = self.increase_pts_if_reshuffle
        if self.alpha >= 1:
            Nbr_f_pts_max_per_batch = int(Nbr_f_pts_max_per_batch *factor )
            Nbr_f_pts_min_per_batch = int(Nbr_f_pts_min_per_batch *factor )
            self.alpha += 1 
            return self.alpha, Nbr_f_pts_max_per_batch, Nbr_f_pts_min_per_batch
    ############################################### 
    def revert_the_increase(self, Nbr_f_pts_max_per_batch, Nbr_f_pts_min_per_batch):
        factor = self.increase_pts_if_reshuffle
        if self.alpha > 1:
            Nbr_f_pts_max_per_batch = int(Nbr_f_pts_max_per_batch / (factor ** (self.alpha - 1)))
            Nbr_f_pts_min_per_batch = int(Nbr_f_pts_min_per_batch / (factor ** (self.alpha - 1)))
            self.alpha = 1  # Reset self.alpha to 1 
            return Nbr_f_pts_max_per_batch, Nbr_f_pts_min_per_batch
        else:
            return Nbr_f_pts_max_per_batch, Nbr_f_pts_min_per_batch
    ###############################################    
    def plot_ini(self,batch_X_ini,batch_phi_ini,X_ini,Nx,Ny,path,t_min,t_max,epoch):
        #print(batch_X_ini.shape,batch_u_ini.shape)
        scatter =plt.scatter(batch_X_ini[:,0], batch_X_ini[:,1], c=batch_phi_ini, marker='*', cmap='jet', s=5, label='IC: initial condition')
        cbar = plt.colorbar(scatter, ax=plt.gca())
        plt.xlim([self.x.min(),self.x.max()])
        plt.ylim([self.y.min(),self.y.max()])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'X_ini_train (n={batch_X_ini.shape[0]})\nFor the training interval: t_min: {t_min:.5f}, t_max: {t_max:.5f}')
        filename = f"IC_epoch_{epoch+1} - For the training interval: t_min: {t_min:.5f}, t_max: {t_max:.5f}.png"
        plt.savefig(os.path.join(path, filename))
        plt.close()
    ##################################################################################################
    ##################################################################################################
    ####################################       Train       ###########################################
    ##################################################################################################
    ##################################################################################################
    def train(self,epochs,batch_size_max,thresh,epoch_scipy_opt=1000,epoch_print=500, epoch_resample=100,\
            initial_check=False,save_reg_int=100,num_train_intervals=10,\
            discrete_resolv=True,fraction_ones_per_int_pts=0.3,fraction_zeros_per_int_pts=0.3,coef_increase_points_f=2,coef_increase_points_ic=2,\
            path=None, path_weights_all_pinns=None,save_weights_pinns=True,\
                communicate_pinns=False,change_pinn_candidate=False,Thresh_Master=0.1,\
                    optimize_master=False,transfer_learning=False,\
                    denoising_loss=False, loss_sum_constraint=False): 

        # time intervals 
        time_subdomains=np.linspace(self.lb[2],self.ub[2],num_train_intervals+1)
        count=0
        #thresh
        self.thresh=thresh
        # init
        X_ini=self.X_ini  # N_ini points (user selection)
        phi_ini=self.phi_ini  
        X_ini_all=self.X_ini_train_all # All ini points (to use by the PINN if mini_batches are not filled)
        phi_ini_all=self.phi_ini_train_all # All ini points (to use by the PINN if mini_batches are not filled)
        
        N_batches=self.N_batches
        thresh_interface=self.thresh_interface

        # loss
        list_loss_workers=[]
        global denoising_loss_
        denoising_loss_=np.copy(denoising_loss)
        global loss_sum_constraint_
        loss_sum_constraint_=np.copy(loss_sum_constraint)

        # dummy params (flags)
        global flag
        flag=0
        flag_weights=1
        flag_shuffle_for_scipy=0
        flag_over_fit=0
        alpha=1
        flag_reload=0
        global flag_reduce_batches
        global flag_scipy
        flag_scipy=False
        flag_reduce_batches=0
        flag_train_Master=0
        flag_print_master=1
        do_master_loop=0
        global count_
        count_=0
        debug_scipy_master=1
        flag_scipy_optimize_pinns= 0 
        count_scipy_iter=0
        dummy_flag=1
        flag_pass=0
        flag_print_ignore=1
        global bool_flag_continuity
        bool_flag_continuity=1 
               
        global path_save
        path_save = os.path.join(os.getcwd(),"save_figs")  
        # shuffle Collocation points
        idx = np.random.permutation(self.X_f.shape[0])
        self.X_f = self.X_f[idx]
  
        # get N_b and N_ini
        N_b=0#self.X_lb.shape[0]
        N_ini=self.X_ini.shape[0]

        path_weights_all_pinns= os.path.join(os.getcwd(),'weights_all_workers_pinns')
        path_get_weights_all_pinns= os.path.join(os.getcwd(),'get_weights_all_workers_pinns')
        
        # Create a dictionary to store the weights for each time interval 
        weights_dict = {}
        weights_key = 0
        # Get current process ID
        pid = os.getpid()
        # Get current process object
        process = psutil.Process(pid)
        with open("usage_log.txt", "w") as f_mem:
            f_mem.write("This is a test.")
            
        file_path = "intercations_file.txt"

        # Check if the file exists before attempting to remove it
        if os.path.exists(file_path):
            os.remove(file_path)
            tf.print(f"File '{file_path}' has been successfully removed.")
        else:
            tf.print(f"File '{file_path}' does not exist.")
            
        with open("intercations_file.txt", "w") as f_int:
            pass 
            global Master_PINN
            Master_PINN = self
            Master_PINN.pinns = self.pinns
            ############################################
            ############### EPOCH LOOP #################
            ############################################
            if N_batches >= 1:  # security step
                global epoch 
                for epoch in range(epochs): 
                    ####################
                    if epoch==0 or flag_reduce_batches==1 or flag :
                        matrix_size = int(np.sqrt(N_batches))
                        matrix = np.arange(N_batches).reshape(matrix_size, matrix_size)
                        quarter1 = matrix[:matrix_size//2, :matrix_size//2].flatten()
                        quarter2 = matrix[:matrix_size//2, matrix_size//2:].flatten()
                        quarter3 = matrix[matrix_size//2:, :matrix_size//2].flatten()
                        quarter4 = matrix[matrix_size//2:, matrix_size//2:].flatten()
                        Quarters=[quarter1,quarter2,quarter3,quarter4]                        
                        #tf.print("\n quarter1:", ", ".join(map(str, quarter1)))
                        #tf.print("quarter2:", ", ".join(map(str, quarter2)))
                        #tf.print("quarter3:", ", ".join(map(str, quarter3)))
                        #tf.print("quarter4:", ", ".join(map(str, quarter4))) 

                        Nf=self.Nf

                        ####################
                        if epoch==0 or flag==1 :  # a new time interval ==> restart (initial number of points, batches, self.pinns)
                            Nbr_f_pts_max_per_batch=np.copy(self.Nbr_f_pts_max_per_batch)
                            Nbr_f_pts_min_per_batch=np.copy(self.Nbr_f_pts_min_per_batch)
                            N_ini_max_per_batch= np.copy(self.N_ini_max_per_batch)
                            N_ini_min_per_batch=np.copy(self.N_ini_min_per_batch)
                            N_batches=int(np.copy(self.N_batches))
                            if epoch>0:  # few epochs of taining are recommended for adam to correctly receive new weights
                                self.re_Initialize_pinns()  


                            #if epoch>0: 
                            #    #tf.print("check: ", flag,flag_train_Master, flag_train_workers ,flag_reduce_batches,N_batches )
                            tf.print("debug:epoch, flag_reduce_batches, flag: ", epoch, flag_reduce_batches, flag)
                            tf.print("\n -------------------------------------------------------------")
                            tf.print("  -----  Epoch: {0:d} <==> N_batches: {1:d}, self.pinns: {2:d}    -------".format(epoch,N_batches,len(self.pinns)))
                            if epoch==0:
                                tf.print("  ----- time domain: ",'t_min: {0:.5f}, t_max: {1:.5f}'.format(self.t_min, time_subdomains[count+1]))
                            else:
                                tf.print(" time domain: ",'t_min: {0:.5f}, t_max: {1:.5f}'.format(t_min, t_max))
                            tf.print("--------------------------------------------------------------\n")

                    ####################    
                    # set time bounds
                    if discrete_resolv:
                        t_min, t_max = time_subdomains[count], time_subdomains[count+1]
                        self.t_min=t_min
                        self.t_max=t_max 
                    else:
                        t_min, t_max = time_subdomains[0], time_subdomains[count+1] 
                        self.t_min=t_min
                        self.t_max=t_max 
                    ####################
                    if count_==0:  
                        if t_min==0 and N_batches==self.N_batches:  
                            if transfer_learning==False: 
                                tf.print("\n !!! no tranfer of learning  !!! \n")     
                            else:  
                                tf.print("\n !!!  Tranfer of learning  !!! \n")                                              
                                self.set_weights_all_pinns(N_batches,path_get_weights_all_pinns) # Initial heritated weights
                                self.test_IC(N_batches,"test_IC")
                        else:
                            if N_batches==self.min_batch_numbers:
                                self.set_weights_all_pinns(N_batches,path_weights_all_pinns) # from previous time interval
                                
                        
                    if flag_reload==1:
                        # Search for the weight file with matching t_min and t_max values
                        weights_dir = 'get_weights/'
                        filename_pattern = f'weights_tmin_{t_min:.5f}_tmax_{t_max:.5f}_*.json'
                        matching_files = glob.glob(weights_dir + filename_pattern)
                        if matching_files:
                            weights_file = sorted(matching_files)[-1]
                            tf.print("t_min, t_max: ",t_min, t_max)
                            tf.print("load weights from: ",weights_file)
                            
                            with open(weights_file, 'r') as f:
                                weights_loaded =json.load(f)['weights']
                            weights_loaded=tf.cast(weights_loaded, dtype=self.precision)
                            self.set_weights(weights_loaded)                    


                    # update selection ( take training points between t_min and t_max)
                    #self.update_selection_for_training_domain_und_bouńdaries(X_f_sub_domain)
                    
                    # **************** X_f for scipy  ***** 
                    # Collocation points to use by scipy optimizer (it is the same as used by Adam but contructed in a different way)                    
                    if epoch==0 or flag==1 or flag_shuffle_for_scipy==1 or flag_reduce_batches==1 :
                        X_f_sub_domain_scipy = []
                        #*************************************                    
                        # **************** IC ****************
                        #*************************************
                        # move the IC points
                        if epoch==0 or flag==1 and flag_shuffle_for_scipy==0:
                            if epoch==0:
                                X_ini_all_sub_domain = X_ini_all
                                phi_ini_all_sub_domain = phi_ini
                                self.X_ini_all_sub_domain=X_ini_all_sub_domain
                                self.phi_ini_all_sub_domain = phi_ini_all_sub_domain

                            if flag_train_Master==0:
                                flag_train_workers=1 # training workers-based until convergence and reaching next time interval

                            if discrete_resolv and epoch>0:
                                    X_ini_all_sub_domain=self.X_ini_all_sub_domain # already updated 
                                    phi_ini_all_sub_domain=self.phi_ini_all_sub_domain
                                    #tf.print(self.X_ini_all_sub_domain)
                        # *********************************************************************
                        # --------------   Prepare IC points  ---------------------------
                        # ---  Note this is not only for scipy but IC poitns for adam too-----
                        # *********************************************************************
                        #**************************
                        #**************************
                        if epoch==0 or flag or flag_reduce_batches==1 or epoch % epoch_resample==0 or flag_shuffle_for_scipy :
                            if count_scipy_iter==0:  
                                num_x_intervals,num_y_intervals=self.set_IC_data_for_pinns(N_batches,fraction_ones_per_int_pts,fraction_zeros_per_int_pts,N_ini_min_per_batch,N_ini_max_per_batch,flag_plot=False)
                            self.set_Collocation_points_for_pinns(N_batches,Nf,flag_plot=False)

                    if epoch % epoch_resample==0 and count_>0:   
                        self.set_Collocation_points_for_pinns(N_batches,Nf, flag_plot=False)
                        
                        #**************************
                        #**************************
                    ####################################################################################################
                    ####################################################################################################
                    ###############  Minibatching   #################################################################### 
                    ####################################################################################################
                    ###################################################################################################
                    ######################################################################################
                    ######################################################################################
                    #if epoch==0 or flag or flag_reduce_batches: 
                    global dict_I
                    dict_I=self.initialize_dict_I(N_batches)  # to compute the multi-phases intercations term(I_alpha -I_beta)
                
                    global dict_I_previous
                    dict_I_previous=self.initialize_dict_I(N_batches)    
                    
                    if epoch==0 or flag or flag_reduce_batches:             
                        quarters = {
                            'quarter1': quarter1,
                            'quarter2': quarter2,
                            'quarter3': quarter3,
                            'quarter4': quarter4
                        }
                        
                        if N_batches==self.N_batches:
                            quarters_max_indices, dict_quarters_infos=self.get_and_store_quarters_max_indices(self.pinns,quarters,N_batches)

                        else : 
                            quarters_max_indices, dict_quarters_infos=self.get_and_store_quarters_max_indices_case_2(self.pinns,quarters,N_batches,dict_quarters_infos)    

                        
                        global interactions
                        interactions , total_interactions=self.Phases_interactions_infos(N_batches)
                        

                        if self.num_phases>1:
                            self.set_phases_interaction( N_batches, interactions)
                            self.plot_interaction(interactions, N_batches)
                            
                        ###################################################
                        tf.print("quarters_max_indices:", ", ".join(map(str, quarters_max_indices)))
                        tf.print("dict_quarters_infos:", dict_quarters_infos)

                        self.plot_IC_data_of_selected_pinns(N_batches,quarters_max_indices) 
                        self.plot_Collocation_data_of_selected_pinns(N_batches,quarters_max_indices,Nf)
                    
                    ######################################################################################
                    ######################################################################################
                    ######################################################################################
                    ######################################################################################
                    ######################################################################################
                    # get common F batches for PDEs loss
                    if epoch % epoch_resample==0 or count_==0 or flag or flag_reduce_batches or flag_shuffle_for_scipy==1: 
                        tf.print(" \n !!! updating Collocation points at epoch ",epoch, "!!! \n" )
                        self.update_batch_Xf()
                    
                    #if count_==0 or flag or flag_reduce_batches or flag_shuffle_for_scipy==1: 
                    #    self.update_batch_X_ini(pinns)
                    #tf.print(stop_here)
                    #########################################################################
                    batch_args = []  # for parallelization
                    pinns_adam = []
                        
                    random_indices = random.sample(range(N_batches), 1) # for check 
                    for pinn_idx in range(len(self.pinns)):
                        pinn= self.pinns[pinn_idx] 
                        #if pinn.flag==0 or (pinn.idx_pinn not in quarters_max_indices): # internal code management, memory release
                        #    g=pinn.evaluate(pinn.batch_Xf)
                        #    pinn.I_beta=tf.zeros_like(g)

                        batch_Xf = []# selected inside loss_ pde 
                        batch_X_ini = pinn.batch_X_ini
                        batch_phi_ini = pinn.batch_phi_ini
                        #batch_X_ini_all=pinn.batch_X_ini_all
                        #batch_phi_ini_all=pinn.batch_phi_ini_all
                        batch_X_lb = [] 
                        batch_X_ub = [] 
                        batch_X_ltb =[] 
                        batch_X_rtb = [] 
                        
                        batch_arg=[batch_Xf, batch_X_ini, batch_X_lb, batch_X_ub,batch_X_ltb,batch_X_rtb, batch_phi_ini,pinn] 
                        
                        if pinn.flag>0 and pinn.idx_pinn in quarters_max_indices : #and pinn.loss_value >self.thresh:
                            pinns_adam.append(pinn)
                            batch_args.append(batch_arg)

                        if epoch >0 and pinn.idx_pinn in quarters_max_indices:
                            pinn_data_for_scipy=[batch_Xf, batch_X_ini, batch_X_lb, batch_X_ub,batch_X_ltb,batch_X_rtb, batch_phi_ini]
                            pinn.pinn_data_for_scipy= pinn_data_for_scipy  # Update the 'self.batch' attribute with 'batch_args'

                        pinn.Nfeval =multiprocessing.Value('i', 0)   # reset (for later scipy optimization)
                        pinn.lr=self.lr   # always update pinns learning rate 

                        
                        #if pinn.flag==0 :# release mem
                        #   del pinn 
                    count_ +=1        
                    
                    ################################################
                    ################################################
                    ############    Parallell  #####################
                    ############## Training   ######################
                    ################################################
                    self.release_memory( quarters_max_indices)
                    ################################################
                    #if len(pinns_adam)>0:
                    #    self.pinns_adam=pinns_adam
                        
                    if flag_train_workers==1 and len(batch_args) >0:                          
                        #tf.print("Epoch: ", epoch,"len(batch_args): ",len(batch_args) )
                        num_processes=min(multiprocessing.cpu_count(),len(batch_args) )# a maximum of  processes are available
                        
                        with tf.device('/CPU:0'):
                            pool = multiprocessing.pool.ThreadPool(processes=num_processes)
                            results = pool.starmap(self.process_batch, batch_args)
                            #gc.collect()
                            pool.close()
                            pool.join()

                            # Accumulate the losses from each batch
                            global_loss_workers = 0.0
                            global_loss_f_workers = 0.0
                            global_loss_IC_workers = 0.0
                            global_loss_BC_workers = 0.0   
                            
                            min_loss = float('inf')
                            max_loss = float('-inf')
                            min_loss_idx = -1
                            max_loss_idx = -1
                            pinns_adam_below_thresh=[]
                            pinns_adam_above_thresh=[]

                            for pinn_idx, (idx_pinn, loss_, loss_BC, loss_IC, loss_f) in enumerate(results):                            
                                pinn =copy.copy(pinns_adam[pinn_idx])
                                
                                global_loss_workers += loss_
                                global_loss_f_workers += loss_f
                                global_loss_IC_workers += loss_IC
                                global_loss_BC_workers += loss_BC
                                self.pinns[pinn.order].loss_value=loss_               
                                if loss_ < min_loss:
                                    min_loss = loss_
                                    min_loss_idx = pinn.idx_pinn
                                if loss_ > max_loss:
                                    max_loss = loss_
                                    max_loss_idx = idx_pinn
    
                                if loss_ < self.thresh :  
                                    pinns_adam_below_thresh.append(pinn.idx_pinn)  

                                if loss_ >= self.thresh : 
                                    pinns_adam_above_thresh.append(pinn.idx_pinn) 
                                del pinn

                            del results
                            """
                            if epoch % epoch_print == 0:
                                tf.print('pinn {0:d} - Epoch: {1:d}, total_loss: {2:.3e}, loss_BC: {3:.3e}, loss_IC: {4:.3e}, loss_f: {5:.3e}\n'.format(
                                batch_idx, epoch, loss, loss_BC, loss_IC, loss_f))
                            """
                                
                        # Compute the average loss for the epoch
                        global_loss_workers /= len(batch_args)
                        global_loss_f_workers /= len(batch_args)
                        global_loss_IC_workers /= len(batch_args)
                        global_loss_BC_workers/= len(batch_args)

                        
                        self.pinns_adam_above_thresh=pinns_adam_above_thresh  # achtung: are just indexes not pinns 
                        idx_batch_above_thresh = [idx_batch for idx_batch in self.pinns_adam_above_thresh]
                        self.pinns_adam_below_thresh = [pinn.idx_pinn for pinn in self.pinns if pinn.idx_pinn not in idx_batch_above_thresh]                          
                    
                    ########################################################################
                    ##############################      END      ###########################
                    ####################     Parrallel training  of pins ###################
                    ########################################################################
                    ########################################################################                       
                        list_loss_workers.append([global_loss_workers,global_loss_BC_workers,global_loss_IC_workers,global_loss_f_workers])                    

                    ########################################################################
                    ########################      MASTER training ########################
                    ########################################################################
                    
                    ########################################################################
                    ########################################################################
                    ##############################      END      ###########################
                    ########################      MASTER training ########################
                    ########################################################################
                    ########################################################################
                    if  epoch==1 or flag==1 or flag_reduce_batches==1 or flag_shuffle_for_scipy :
                        flag=0  # very important : all data are now prepared for minibatching / training ==> drop the flag 
                        flag_reduce_batches=0
                        flag_shuffle_for_scipy=0

                    ####################################################################################################
                    ####################################################################################################
                    ###############  End Minibatching   ################################################################
                    ####################################################################################################
                    ####################################################################################################              

                    # print losses
                    if epoch % epoch_print == 0:    
                        tf.print('\n==> Epoch: {0:d}, Mean_loss of pinns: {1:.3e}, loss_BC: {2:.3e}, loss_IC: {3:.3e}, loss_f: {4:.3e}'.format(epoch, global_loss_workers, global_loss_BC_workers,global_loss_IC_workers, global_loss_f_workers))
                        tf.print(' => minimum loss: {0:.3e}, corresponding pinn/batch index: {1}'.format(min_loss, min_loss_idx))
                        tf.print(' => maximum loss: {0:.3e}, corresponding pinn/batch  index: {1}'.format(max_loss, max_loss_idx))
                        if count_>= 0 and (len(batch_args)>0):
                            tf.print(" => In : Number of pinns to train: {0:d}".format(len(batch_args))) 
                            if epoch>0:
                                num_pinns_above_thresh = len(self.pinns_adam_above_thresh) 
                                tf.print(" => Out : Number of pinns above Threshold: {0:d}".format(num_pinns_above_thresh)) 
                                tf.print("pinns above Threshold:", ", ".join(map(str, self.pinns_adam_above_thresh)))

                    if epoch % epoch_print == 0 and flag_train_Master==1 and flag_train_workers==0: 
                        tf.print('\n Epoch: {0:d}, total_loss: {1:.3e}, loss_BC: {2:.3e}, loss_IC: {3:.3e}, loss_f: {4:.3e} \n'.format(epoch, global_loss_Master, global_loss_BC_Master,global_loss_IC_Master, global_loss_f_Master))
                    ##################################################
                    ##################################################
                    ###############  Scipy Optimizer pinns  ##########
                    ##################################################
                    ##################################################  
                    
                    #sorted_pinns = sorted(pinns, key=lambda pinn: ( int(pinn.idx_batch)))
                    #for pinn in sorted_pinns:
                    #    tf.print(pinn.idx_phase, pinn.idx_batch, pinn.idx_pinn)
        
                    # call scipy optimizer if loss > thresh
                    if count_ % epoch_scipy_opt == 0 and max_loss > self.thresh and epoch>0 and flag_train_workers==1 and len(pinns_adam)>0 and count_>20: #:and self.f==1: #and N_batches==self.N_batches: 
                        flag_scipy= True
                        tf.print("\n !!! Scipy optimize: !!! - Epoch: ",str(epoch))
                        tf.print("\n ! Scipy iteration number: ",str(count_scipy_iter+1))
                        #global Nfeval
                        Nfeval=1  # reinitialize the global value of Nfeval
                        global list_loss_scipy
                        
                        batch_args_scipy= []
                        pinns_to_optimize=[]
                        for pinn_idx in range(len(self.pinns)):
                            pinn=self.pinns[pinn_idx]
                            pinn.list_loss_scipy = []
                            if pinn.flag==1 and pinn.idx_pinn in pinns_adam_above_thresh:
                                batch_args_scipy.append([pinn,pinn.get_weights().numpy()])
                                pinns_to_optimize.append(copy.copy(pinn))
                            
                        processes=len(batch_args_scipy)

                        # Parallel optimization
                        num_processes_scipy = min(16, processes) # maximum number of cores for parallelization  (max of 16 by security)
                        tf.print("Processes: ", num_processes_scipy)
                        tf.print("Number of pinns to optimize weights: ", len(pinns_to_optimize))
                        pinns_to_optimize_indexes = [str(pinn.idx_pinn) for pinn in pinns_to_optimize]
                        tf.print(" !!! pinns to optimize ==>", ", ".join(pinns_to_optimize_indexes))
                        tf.print("\n")
                        pool_scipy = multiprocessing.pool.ThreadPool(processes=num_processes_scipy)
                        results = pool_scipy.starmap(self.optimize_single_pinn, batch_args_scipy)
                        #gc.collect()
                        pool_scipy.close()
                        pool_scipy.join()
                        
                        del  batch_args_scipy
                    
                        # set optmized weights for each pinn and compute global_loss 
                        global_loss_workers = 0.0
                        global_loss_f_workers = 0.0
                        global_loss_IC_workers = 0.0
                        global_loss_BC_workers = 0.0
                        
                        min_loss = float('inf')
                        max_loss = float('-inf')
                        min_loss_idx = -1
                        max_loss_idx = -1
                        list_of_losses = []

                        # Loop over PINNs and compute global_loss
                        global_loss_workers = 0.0
                        global_loss_f_workers = 0.0
                        global_loss_IC_workers = 0.0
                        global_loss_BC_workers = 0.0

                        for _, (pinn, result) in enumerate(zip(pinns_to_optimize, results)):
                            pinn_idx=pinn.idx_pinn
                            pinn.set_weights(result.x)
                            # to make sure pinn is pinns is updated
                            self.find_pinn_by_idx_pinn(pinn.idx_pinn, target_pinns=None).set_weights(result.x)
                            loss,loss_BC,loss_IC,loss_f=  pinn.list_loss_scipy[-1][0], pinn.list_loss_scipy[-1][1], pinn.list_loss_scipy[-1][2], pinn.list_loss_scipy[-1][3]
                            list_of_losses.append(loss)
                            self.pinns[pinn.order].loss_value=loss
                            global_loss_workers += loss
                            global_loss_f_workers += loss_f
                            global_loss_IC_workers += loss_IC
                            global_loss_BC_workers += loss_BC
                            
                            if loss < min_loss:
                                min_loss = loss
                                min_loss_idx = pinn_idx
                            if loss > max_loss:
                                max_loss = loss
                                max_loss_idx = pinn_idx

                        global_loss_workers /= len(list_of_losses) 
                        global_loss_f_workers /= len(list_of_losses) 
                        global_loss_IC_workers /= len(list_of_losses) 
                        global_loss_BC_workers/= len(list_of_losses) 
                        
                        pinns_below_avg = []
                        pinns_above_avg = []
                        for idx, loss in enumerate(list_of_losses):
                            pinn_idx=pinns_to_optimize[idx].idx_pinn
                            pinn = pinns_to_optimize[idx]  
                        
                            if loss < self.thresh:
                                pinns_below_avg.append(pinn_idx)
                            else:
                                pinns_above_avg.append(pinn_idx)  

                        tf.print("\n !!! number of pinns above the Treshhold: ", len(pinns_above_avg)) 
                        tf.print("\n !!! pinns above the Treshhold ==>", ", ".join(map(str, pinns_above_avg)))
                        tf.print("\n !!! Scipy optimization done !!!\n ")
                        flag_scipy_optimize_pinns=1
                        
                        tf.print('==> loss after L-BFGS-B optimization for Epoch: {0:d}, Mean_loss of pinns: {1:.3e}, loss_BC: {2:.3e}, loss_IC: {3:.3e}, loss_f: {4:.3e}'.format(epoch, global_loss_workers,global_loss_BC_workers,global_loss_IC_workers, global_loss_f_workers))
                        tf.print(' => minimum loss: {0:.3e}, corresponding pinn index: {1:}'.format(min_loss, min_loss_idx))
                        tf.print(' => maximum loss: {0:.3e}, corresponding pinn  index: {1:}'.format(max_loss, max_loss_idx))

                        flag_scipy=False
                        if max_loss > self.thresh: 
                            count_scipy_iter+=1
                            flag_shuffle_for_scipy=1
                        
                        # pinn(s) still not converged !!!  
                        # ====>  Get weights from nearest neighbours ==> one more chance to converge
                        idx_pinn_above_avg = [idx_pinn for idx_pinn in pinns_above_avg]
                        pinns_below_thresh = [pinn for pinn in self.pinns if pinn.idx_pinn not in idx_pinn_above_avg]  
                        pinns_above_thresh = [pinn for pinn in self.pinns if pinn.idx_pinn in idx_pinn_above_avg]                           
                        if count_scipy_iter == 30 and len(pinns_above_avg) > 0 and len(pinns_below_thresh) > 0:
                            #tf.print("\n pinns_below_avg: ", [pinn.idx_pinn for pinn in pinns_below_thresh])
                            #tf.print("\n pinns_above_thresh: ", [pinn.idx_pinn for pinn in pinns_above_thresh])
                            tf.print("\n !!! number of scipy iterations: ", count_scipy_iter)
                            for pinn in pinns_above_thresh:
                                # Filter the list_below_thresh to include only pinns with the same idx_phase
                                filtered_pinns_below_thresh = [pinn_below for pinn_below in pinns_below_thresh if pinn_below.idx_phase == pinn.idx_phase]
                                #tf.print("\n pinn: ", pinn.idx_pinn, "filtered_pinns_below_thresh: ", [pinn.idx_pinn for pinn in filtered_pinns_below_thresh])
                                if len(filtered_pinns_below_thresh)>0:
                                    differences = [abs(int(pinn.idx_batch) - int(pinn_below.idx_batch)) for pinn_below in filtered_pinns_below_thresh]
                                    min_diff=min(differences)
                                    selected_pinn = next((pinn_below for pinn_below in filtered_pinns_below_thresh if abs(int(pinn_below.idx_batch) - int(pinn.idx_batch)) == min_diff), None)

                                    if selected_pinn is not None:
                                        tf.print("\n !!! ==> get for pinn: ", pinn.idx_pinn, " weights of the  pinn:", selected_pinn.idx_pinn)
                                        pinn.set_weights(selected_pinn.get_weights())
                                    else:
                                        tf.print("No suitable pinn found for pinn ", pinn.idx_pinn)
                                    del filtered_pinns_below_thresh

                        ##############################
                        del results, pinns_below_thresh, pinns_above_thresh
                    ##################################################
                    ##################################################
                    ###############  Scipy Optimizer  Master  ########
                    ##################################################
                    ##################################################  

                    ########################################################################
                    ########################################################################
                    if flag_train_workers==1 and flag_print_ignore==1: 
                        thresh_loss=np.copy(max_loss)
                    elif flag_train_Master==1 and flag_train_workers==0 : 
                        thresh_loss=np.copy(global_loss_Master)
                        
                    if thresh_loss < self.thresh and  epoch % epoch_print == 0 and flag_train_workers==0 :
                        Nbr_f_pts_max_per_batch,Nbr_f_pts_min_per_batch=self.revert_the_increase(Nbr_f_pts_max_per_batch, Nbr_f_pts_min_per_batch)

                    ########################################################################
                    ########################################################################
                    ###############  Save and change Domain (time or N:batches ) ###########
                    ########################################################################
                    ########################################################################                                           
                    if thresh_loss < self.thresh and t_max<=self.ub[2] and flag==0 and epoch>0 and epoch % 17==0 and self.f==1: #the last is added to ensure that the model train a bit before new changes and continuity of the minbatching   
                        
                        if flag_train_workers==1: 
                            tf.print("\n max_loss: {0:.3e} < Threshold: {1:.3e}\n".format(thresh_loss, self.thresh))
                            count_=0 # reset 
                            count_scipy_iter=0 # reset 
                            flag_print_ignore=1 # reset
                            self.coef_reduction*=1.5
                            
                        ################################################################
                        if  flag_weights:  # save weights at each time-domain change 
                            Nbr_f_pts_max_per_batch,Nbr_f_pts_min_per_batch=self.revert_the_increase(Nbr_f_pts_max_per_batch, Nbr_f_pts_min_per_batch)  
                            # save predictions 
                            ################################
                            if discrete_resolv: 

                                if flag_train_workers==1:  # workers pinns
                                    if  N_batches==self.min_batch_numbers:  #N_batches==self.N_batches or
                                        self.save_weights_for_pinns(N_batches, path_weights_all_pinns, t_min, t_max,quarters_max_indices)                                     
                                    if  N_batches==self.min_batch_numbers or N_batches==self.N_batches: #N_batches==self.min_batch_numbers or
                                        tf.print("remind ,  N_batches " , N_batches)
                                        self.save_predictions_discret_workers_Master(epoch,path,path_weights_all_pinns,self.X_phi_test,\
                                                X_ini,phi_ini,N_b,t_min, t_max,N_batches,flag_Master=False)

                                        #tf.print(stop_here)
                                    # # will be activated again in the next time interval
                                    flag_reduce_batches=1  # reduce batches  
                                    flag_scipy_optimize_pinns= 0 # re-ini
                                    flag_shuffle_for_scipy=0
                                    #tf.print(stop_here)
                                ################################      
                                if flag_reduce_batches==1:
                                    # reduce number of batches 
                                    N_batches=int((np.sqrt(N_batches)-2)**2)

                                    if N_batches>1:
                                        tf.print('==> Epoch: {0:d}, Mean_loss of pinns: {1:.3e}, loss_BC: {2:.3e},\
                                            loss_IC: {3:.3e}, loss_f: {4:.3e}'.format(epoch, global_loss_workers,global_loss_BC_workers,global_loss_IC_workers, global_loss_f_workers))
                                        tf.print("\n reduce N_batches from Epoch: {0:d} ==>  N_batches: {1:d}\n".format(epoch+1,N_batches))
                                        tf.print("\n max_loss: {0:.3e}, min_loss: {1:.3e}\n".format(max_loss, min_loss))
                                        # update the list of pinns basing on a Pyramid-like or a Convolutional- like selection    
                                        Nbr_f_pts_max_per_batch, Nbr_f_pts_min_per_batch, N_ini_max_per_batch, N_ini_min_per_batch = self.reduce_batches_Pyramid_like_training(
                                            N_batches, num_x_intervals, num_y_intervals, path_weights_all_pinns, coef_increase_points_f, coef_increase_points_ic,
                                            Nbr_f_pts_max_per_batch, Nbr_f_pts_min_per_batch, N_ini_max_per_batch, N_ini_min_per_batch,
                                            quarters_max_indices, dict_quarters_infos)

                                    elif N_batches < self.min_batch_numbers:  
                                        flag_train_workers=0
                                        flag_reduce_batches=0 
                                        flag_shuffle_for_scipy=0
                                        N_batches=self.min_batch_numbers
                                        flag_pass=1
                                        count_scipy_iter=0 # reset  

                                        #for pinn_idx in range(len(self.pinns)):
                                        #    tf.print("pinn.idx_pinn ",pinn_idx, self.pinns[pinn_idx].idx_pinn)

                                        tf.print("\n pinns to pinns Prediction ===> next time interval " )
                                                                                
                                        #f discrete_resolv and epoch>0:
                                        #   self.get_data_next_time_interval(N_batches)          
                            ################################    
                            else: # continuous resolv 
                                self.save_predictions_continous(epoch,path,self.X_phi_test,\
                                    X_ini,phi_ini,N_b,t_min, t_max)

                            if t_max==self.ub[2]:   # stop saving weigthts (alles abgeschlossen)
                                self.thresh=global_loss_Master/1.1
                                tf.print("Now optimizing the solution for the new threshold: {:.3e}".format(self.thresh))
                                #flag_weights=0                            

                        # Save Master PREDICTIONS and prepare training on the next time domain
                        if t_max<self.ub[2] and N_batches==self.min_batch_numbers and flag_pass==1 and epoch % 17==0 :
                            
                            tf.print('Increase time interval ==> Epoch: {0:d}, total_loss: {1:.3e}, loss_BC: {2:.3e}, loss_IC: {3:.3e}, loss_f: {4:.3e}'.format(epoch, global_loss_workers, global_loss_BC_workers,global_loss_IC_workers, global_loss_f_workers))
                            tf.print("\n ")
                            #tf.print(stop_here)
                            count+=1
                            # set new/next time bounds
                            if discrete_resolv:
                                t_min, t_max =time_subdomains[count], time_subdomains[count+1] 
                                self.t_min=t_min
                                self.t_max=t_max 
                            else:
                                t_min, t_max =time_subdomains[0], time_subdomains[count+1] 
                                self.t_min=t_min
                                self.t_max=t_max 
                            tf.print("\n ===================================================> \n ")
                            tf.print("  Change the time domain to: ",'t_min: {0:.5f}, t_max: {1:.5f}'.format(t_min, t_max))
                            tf.print("\n ===================================================> \n ") 
                            flag=1    
                            flag_pass=0
                            flag_reduce_batches=0
                            N_batches=self.N_batches
                            count_=0

                            #tf.print(stop_here)

        f_mem.close()
        f_int.close()
        return list_loss_workers
    ###############################################
