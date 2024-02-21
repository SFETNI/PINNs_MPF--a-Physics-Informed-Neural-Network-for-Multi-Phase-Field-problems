
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
        
        self.t_min=0
        self.t_max=1   
        
        self.f=1
        self.ic=1
        self.bc=1   
        self.scipy_max_iter=1500
        self.alpha=1 # to increase scipy iterations when reshuffling 
        self.lr=0.00001 #0.000001
        self.precision=tf.float64
        self.precision_="float64"
        self.thresh=0
        self.increase_pts_if_reshuffle=1.1
        
        self.N_batches=np.copy(N_batches) # Total number of batches, default initialization 
        self.min_batch_numbers=4
        self.Nbr_f_pts_max_per_batch=np.copy(Nbr_f_pts_max_per_batch)# maximum number  Collocation points per batch for Scipy optimizer 
        self.Nbr_f_pts_min_per_batch = np.copy(Nbr_f_pts_min_per_batch)# minimum number of Collocation points per batch for Scipy optimizer 
        self.N_ini_max_per_batch=np.copy(N_ini_max_per_batch) #maximum number of IC points per batch for Scipy optimizer 
        self.N_ini_min_per_batch=np.copy(N_ini_min_per_batch)
        self.Nfeval_master=multiprocessing.Value('i', 1)
        
        # workers pinns  
        self.pinns=[]
        self.pinn_data_for_scipy = []# minibatches for each pinn for scipy optimizer 
        self.batch_Xf_for_pred= tf.Variable([], dtype=self.precision, trainable=False)
        self.limits=[]
        self.idx_batch=multiprocessing.Value('i', 1000)  # to be accessed in multiprocessing
        self.flag=0
        self.list_loss_scipy = []
        self.Nfeval = multiprocessing.Value('i', 0) 
        self.pinns_adam=[]
        self.pinns_adam_above_thresh=[]
        self.pinns_adam_below_thresh=[]
        
        self.reseve_weights= []  # each worker pinn will get a reserve weights (from the second level of the PYRAMID)
        # ==> to use to converge
       
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
        lb = tf.reshape(self.lb[:2], (1, -1))
        lb = tf.cast(self.lb[:2], self.precision)
        ub = tf.reshape(self.ub[:2], (1, -1))
        ub = tf.cast(self.ub[:2], self.precision)
        #tf.print("here in evaluate In X: ", X.shape)
        H =X # (X - lb) / (ub - lb)
        #tf.print("here in evaluate In: ", H.shape)
        #tf.print("H.min_max: ",tf.reduce_min(H), tf.reduce_max(H))
        for l in range(0,len(self.layers)-2):
            W = self.W[2*l]
            b = self.W[2*l+1]
            H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))

        W = self.W[-2]
        b = self.W[-1]
        Y = tf.math.add(tf.matmul(H, W), b) # For regression, no activation to last layer
        Y = tf.nn.sigmoid(Y) # apply sigmoid activation function
        del lb, ub, H, X, W,b
        #tf.print("here in evaluate Out: ", Y.shape)
        return Y
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
    def Initialize_pinns(self,path_weights_all_pinns):
        pinns=self.re_Initialize_pinns(flag_epoch_0=True)
        for pinn_idx in range(self.N_batches):
            pinn_path_weights = os.path.join(path_weights_all_pinns, f"pinn_{pinn_idx}")
            if not os.path.isdir(pinn_path_weights):
                os.mkdir(pinn_path_weights)
        return pinns 
    ############################################### 
    def re_Initialize_pinns(self, flag_epoch_0=False):
        if flag_epoch_0:
            tf.print("\n ! Initilization of all workers pinns \n")
        else:
            tf.print("\n ! Re-Initilization of all workers pinns \n")
        pinns = []  # List to store the PINN objects
        for _ in range(self.N_batches):
            pinn = Sequentialmodel(layers=self.layers, X_f_train=self.X_f, X_ini_train=self.X_ini,\
                                    phases_ini_indexes=self.phases_ini_indexes,all_ini_flags_matrix=self.All_flag_ini,\
                                    Phi_ini=self.All_phi_ini,phi_ini_train=self.phi_ini, N_ini=self.N_ini,X_phi_test=self.X_phi_test,\
                                    X_ini_train_all=self.X_ini_train_all, phi_ini_train_all=self.phi_ini_train_all,\
                                        all_interfaces=self.All_interfaces_ini,\
                                    X_lb_train=self.X_lb, X_ub_train=self.X_rtb,\
                                    X_ltb_train=self.X_ltb, X_rtb_train=self.X_rtb,\
                                    X=None,Y=None,T=None,x=self.x,y=self.y,lb=self.lb, ub=self.ub, mu=self.mu, sigma=self.sigma, delta_g=self.delta_g,\
                                        eta=self.eta,Nx=self.Nx,Ny=self.Ny,Nt=self.Nt,phi_sol=None,pinns =self.pinns,num_phases=self.num_phases,
                                    N_batches=self.N_batches,\
                                    Nbr_f_pts_max_per_batch=self.Nbr_f_pts_max_per_batch,\
                                    Nbr_f_pts_min_per_batch=self.Nbr_f_pts_min_per_batch,\
                                    N_ini_max_per_batch=self.N_ini_max_per_batch,\
                                    N_ini_min_per_batch=self.N_ini_min_per_batch)   
                            
            #pinn.set_weights(weights_loaded)  # Inherit weights from the master PINN
            pinn.set_weights(self.get_weights())
            pinns.append(pinn) 
            
        self.pinns=pinns 
        return pinns       
    ###############################################     
    def test_IC(self,N_batches,pathOutput):

        # sublimits
        sublimits_list = []
        num_x_intervals = int(np.ceil(np.sqrt(N_batches)))
        num_y_intervals = int(np.ceil(np.sqrt(N_batches)))
        step_x = max(self.x) / num_x_intervals
        step_y = max(self.y) / num_x_intervals
        for i in range(num_x_intervals):
            for j in range(num_y_intervals):
                x_min = i * step_x
                x_max = (i + 1) * step_x
                y_min = j * step_y
                y_max = (j + 1) * step_y
                sublimits_list.append([x_min, x_max, y_min, y_max])
                
        # predcitions 
        for phase_idx in range(self.num_phases):
            n=len(self.X_ini_train_all)
            N=self.num_phases
            X_phi_test= self.X_ini_train_all[int(n*phase_idx/N)+1:int(n*(phase_idx+1)/N)]
            phi_test= self.phi_ini_train_all[int(n*phase_idx/N)+1:int(n*(phase_idx+1)/N)]
            
            phi_evolution_t_min = []
            phi_evolution_t_max = []
            X_phi_test_subsets = []
            # Predict for each phase
            for i in range(N_batches):
                pinn=self.pinns[i]
                
                t_min, t_max = self.t_min, self.t_max # self.PRE_POST.extract_t_min_t_max(weights_file)
                x_min, x_max, y_min, y_max=sublimits_list[i]

                indices = np.where(
                    (X_phi_test[:, 0] >= x_min) &
                    (X_phi_test[:, 0] <= x_max) &
                    (X_phi_test[:, 1] >= y_min) &
                    (X_phi_test[:, 1] <= y_max)
                )
                
                X_phi_test_sub = X_phi_test[indices] 
                phi_test_sub = phi_test[indices]
                            
                phi_pred_t_min  = pinn.evaluate(X_phi_test_sub)
                
                phi=phi_pred_t_min
                plt.scatter(X_phi_test_sub[:, 0], X_phi_test_sub[:, 1], cmap=plt.get_cmap('viridis'), c=phi,vmin=0,vmax=1)
                x_avg = (x_min + x_max) / 2
                y_avg = (y_min + y_max) / 2

                plt.text(x_avg, y_avg, f"{i}", color='black', ha='right', va='bottom')
                
                #plt.show()
                #plt.xlim([x_min, x_max])
                #plt.ylim([y_min, y_max])
            plt.colorbar()
            #plt.tight_layout()
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

        #x, y, t, f_0, f_1, f_2, f_3 = tf.split(x_ini, num_or_size_splits=7, axis=1)
        #f_sum=f_0+ f_1+ f_2+ f_3 
        #x_ini = tf.concat([x,y, t], axis=1)

        phi_ini_pred=self.evaluate(x_ini)  
 
        # Reconstruct phi_ini_pred with the modified phase_2
        #reconstructed_phi_ini_pred = tf.stack([phase_0, phase_1, phase_2, phase_3], axis=1)

        MSE_loss_IC = tf.reduce_mean(tf.square(phi_ini-phi_ini_pred))
        #sum_constraint_loss_IC = tf.reduce_mean(tf.square(tf.reduce_sum(phi_ini_pred, axis=1) - 1.0))

        exclude_batch = tf.reduce_any(tf.logical_and(phi_ini > 0, phi_ini < 1))
        if exclude_batch:
            alpha = 1.0 
        else:
            alpha =0.1# 0.0001  
           
        beta = 0   # Weight for the sum constraint loss
        gamma=0   # Weight for the sum_constraint_interfaces_IC loss
        
        loss_IC = alpha * MSE_loss_IC #+ beta * sum_constraint_loss_IC #+ gamma* sum_constraint_interfaces_IC
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
    def Sigma(self,phi_alpha,phi_phi_beta):
        return self.sigma # we assume the same interfacial energies for all phases
    ###############################################       
    def I_phi(self,lap_phi_alpha,phi_alpha,Prefactor):
        return (lap_phi_alpha +Prefactor*phi_alpha )
    ###############################################
    #@tf.function
    def loss_PDE(self, X_f, phi_ini):
        g = tf.Variable(X_f, dtype=self.precision, trainable=False)
        
        x, y, t, f_0, f_1, f_2, f_3 = tf.split(g, num_or_size_splits=7, axis=1)
        Prefactor = np.pi**2 / self.eta**2
    
        take_batch = tf.reduce_any(tf.logical_and(phi_ini > 0, phi_ini < 1))
    
        if not take_batch:
            return tf.constant(0.0, dtype=self.precision)
    
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(t)
            g = tf.concat([x,y, t,f_0,f_1,f_2,f_3], axis=1)
            phase_fields = self.evaluate(g)
            
            loss_f = 0.0
            phase_losses = []
            for alpha in range(self.num_phases):                
                phi_alpha = phase_fields[:, alpha:alpha+1]
                dPsi_dt=np.zeros_like(phi_alpha )
                right_side_eqn = tf.zeros_like(phi_alpha)
                tape.watch(phi_alpha)
                phi_x_alpha = tape.gradient(phi_alpha, x)     
                tape.watch(phi_x_alpha)              
                phi_y_alpha = tape.gradient(phi_alpha, y)
                tape.watch(phi_y_alpha)  
                phi_t_alpha = tape.gradient(phi_alpha, t)
                phi_xx_alpha = tape.gradient(phi_x_alpha, x)                   
                phi_yy_alpha = tape.gradient(phi_y_alpha, y)
                lap_phi_alpha = phi_xx_alpha + phi_yy_alpha
                
                for beta in range(self.num_phases):
                    if alpha != beta:
                        phi_beta = phase_fields[:, beta:beta+1]  
                        tape.watch(phi_beta)
                        phi_x_beta = tape.gradient(phi_beta, x)
                        tape.watch(phi_x_beta)
                        phi_y_beta = tape.gradient(phi_beta, y)
                        tape.watch(phi_y_beta)
                        phi_xx_beta = tape.gradient(phi_x_beta, x)
                        phi_yy_beta = tape.gradient(phi_y_beta, y)
                        lap_phi_beta = phi_xx_beta + phi_yy_beta
                        I_alpha=self.I_phi(lap_phi_alpha,phi_alpha,Prefactor)
                        I_beta=self.I_phi(lap_phi_beta,phi_beta,Prefactor)
                        sigma_ab = self.Sigma(phase_fields[alpha], phase_fields[beta])
                        # get dPsi_dt
                        dPsi_dt += sigma_ab * (I_alpha - I_beta) +np.pi**2/(4*self.eta) *self.delta_g
               
                right_side_eqn=   (self.mu/self.num_phases) * dPsi_dt
                phi_t = phi_t_alpha  
                f = phi_t - right_side_eqn
                loss_f += tf.reduce_mean(tf.square(f))
                phase_losses.append(loss_f)
                loss_total = loss_f #+ self.reg_param * tf.reduce_sum(phase_losses)

        del tape,g,x,y,t,dPsi_dt, phase_fields,phi_alpha,phi_t,right_side_eqn, phi_beta, I_alpha, I_beta,phi_x_alpha,phi_y_alpha,phi_x_beta, phi_y_beta,lap_phi_alpha, lap_phi_beta, phi_xx_beta,phi_xx_alpha,phi_yy_beta

        return loss_total
    ###############################################
    def loss(self,xf,x_ini,x_lb,x_ub,x_ltb,x_rtb,phi_ini,abs_x_min,abs_x_max,abs_y_min,abs_y_max):
        loss_IC = self.loss_IC(x_ini,phi_ini)      
        loss_f = 0# self.loss_PDE(xf,phi_ini)        
        loss_BC = 0 #self.loss_BC(x_lb,x_ub,x_ltb,x_rtb,abs_x_min,abs_x_max,abs_y_min,abs_y_max)        
        loss =  self.f*loss_f +self.ic*loss_IC+self.bc*loss_BC #    
        del xf,x_ini,x_lb,x_ub,x_ltb,x_rtb,phi_ini
        return loss, loss_BC,loss_IC, loss_f # loss_BC,loss_IC, loss_f
    ###############################################
    def optimizerfunc(self,parameters):  # Global Optiöization (Master pinn)
        global list_loss_scipy
        #global Nfeval
        self.set_weights(parameters)
               
        
        
        for pinn in self.pinns:
            X_f = np.concatenate([pinn.pinn_data_for_scipy[0] for pinn in self.pinns], axis=0)
            X_ini = np.concatenate([pinn.pinn_data_for_scipy[1] for pinn in self.pinns], axis=0)
            X_lb = np.concatenate([pinn.pinn_data_for_scipy[2] for pinn in self.pinns], axis=0)
            X_ub = np.concatenate([pinn.pinn_data_for_scipy[3] for pinn in self.pinns], axis=0)
            X_ltb = np.concatenate([pinn.pinn_data_for_scipy[4] for pinn in self.pinns], axis=0)
            X_rtb = np.concatenate([pinn.pinn_data_for_scipy[5] for pinn in self.pinns], axis=0)
            phi_ini = np.concatenate([pinn.pinn_data_for_scipy[6] for pinn in self.pinns], axis=0)
   
        # Convert the concatenated arrays to TensorFlow tensors
        X_f = tf.convert_to_tensor(X_f, dtype=tf.float64)
        X_ini = tf.convert_to_tensor(X_ini, dtype=tf.float64)
        X_lb = tf.convert_to_tensor(X_lb, dtype=tf.float64)
        X_ub = tf.convert_to_tensor(X_ub, dtype=tf.float64)
        X_ltb = tf.convert_to_tensor(X_ltb, dtype=tf.float64)
        X_rtb = tf.convert_to_tensor(X_rtb, dtype=tf.float64)
        phi_ini = tf.convert_to_tensor(phi_ini, dtype=tf.float64)   
        
        
        correction_ratio = 40
        num_points_X_ini = X_f.shape[0] // correction_ratio
        selected_indices = np.random.choice(X_ini.shape[0], num_points_X_ini, replace=False)
        selected_indices_tensor = tf.constant(selected_indices, dtype=tf.int32)
        X_ini= tf.gather(X_ini, selected_indices_tensor)
        phi_ini= tf.gather(phi_ini, selected_indices_tensor)
        
        """
        tf.print("X_f shape:", X_f.shape)
        tf.print("X_ini shape:", X_ini.shape)
        tf.print("X_lb shape:", X_lb.shape)
        tf.print("X_ub shape:", X_ub.shape)
        tf.print("X_ltb shape:", X_ltb.shape)
        tf.print("X_rtb shape:", X_rtb.shape)
        tf.print("phi_ini shape:", phi_ini.shape)
        
        plt.scatter(X_ini[:, 0], X_ini[:, 1], c=phi_ini, cmap='jet', s=20)
        plt.show()
        """
        
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
        global list_loss_scipy

        # Extract numeric values from TensorFlow tensors
        total_loss = float(list_loss_scipy[-1][0])
        loss_BC = float(list_loss_scipy[-1][1])
        loss_IC = float(list_loss_scipy[-1][2])
        loss_f = float(list_loss_scipy[-1][3])
        
        if self.Nfeval_master.value % 50 == 0:
            formatted_str = 'Iter: {:d}, total_loss: {:.3e}, loss_BC: {:.3e}, loss_IC: {:.3e}, loss_f: {:.3e}'.format(self.Nfeval_master.value, total_loss, loss_BC, loss_IC, loss_f)
            tf.print(formatted_str)

        self.Nfeval_master.value += 1 
        return list_loss_scipy
    ###############################################
    def optimize_single_pinn(self, pinn, init_params):
        try:
            # Perform the optimization for a single PINN
            import functools
            func_to_minimize = functools.partial(pinn.optimizerfunc_pinn)
            #tf.print(f"pinn {pinn.idx_batch}, maxfun  {pinn.scipy_max_iter}, maxiter {pinn.scipy_max_iter}, self.Nfeval  {self.Nfeval.value}  ")
            result = scipy.optimize.minimize(fun=func_to_minimize,
                                    x0=init_params,
                                    args=(),
                                    method='L-BFGS-B',
                                    jac=True,
                                    callback=pinn.optimizer_callback,
                                    options={'disp': None,
                                                #'maxcor': 5000, 
                                                'ftol': 1 * np.finfo(float).eps,
                                                #'gtol': 5e-8,
                                                #'maxfun':  self.alpha*pinn.scipy_max_iter,
                                                'maxiter':  self.alpha*pinn.scipy_max_iter,
                                                'iprint': -1,
                                                'maxls': 50})
            del func_to_minimize  
            return result
        except Exception as e:
            tf.print(f"An exception occurred for pinn {pinn.idx_batch}")
            tf.print(traceback.format_exc())    
                                                                                            
        return None
    ###############################################
    def optimizerfunc_pinn(self,parameters):
        self.set_weights(parameters)
        
        X_f, X_ini_all, X_lb, X_ub, X_ltb, X_rtb, phi_ini_all = self.pinn_data_for_scipy

        num_samples = int(len(X_ini_all) * 1) # coef if needed 
        if num_samples==0:
            tf.print(" !!! Increase IC points for pinn {0}, num_samples = {1}, len(X_ini_all) = {2} !!!".format(self.idx_batch, num_samples,len(X_ini_all)))

        indices_ini = np.random.choice(len(X_ini_all), size=num_samples, replace=False)
        X_ini=X_ini_all[indices_ini]
        phi_ini=phi_ini_all[indices_ini]
        
        X_f = tf.convert_to_tensor(X_f, dtype=self.precision)
        X_lb = tf.convert_to_tensor(X_lb, dtype=self.precision)
        X_ub = tf.convert_to_tensor(X_ub, dtype=self.precision)
        X_ltb = tf.convert_to_tensor(X_ltb, dtype=self.precision)
        X_rtb = tf.convert_to_tensor(X_rtb,dtype=self.precision)
        X_ini_all = tf.convert_to_tensor(X_ini_all, dtype=self.precision)
        phi_ini_all = tf.convert_to_tensor(phi_ini_all, dtype=self.precision)
        X_ini= tf.convert_to_tensor(X_ini, dtype=self.precision)
        phi_ini= tf.convert_to_tensor(phi_ini, dtype=self.precision)
 
        # Debug 
        """
        X_ini_all = X_ini_all.numpy()
        phi_ini_all = phi_ini_all.numpy()
        phi_values = phi_ini_all[:, 2]
        fig_in = plt.figure()
        ax_in = fig_in.add_subplot(111)
        ax_in.scatter(X_ini_all[:, 0], X_ini_all[:, 1], cmap=plt.get_cmap('viridis'), c=phi_values, vmin=0, vmax=1)
        title = "check_scipy_batch_" + str(self.idx_batch)
        ax_in.set_title(title)
        ax_in.set_xlim(0, 1)
        ax_in.set_ylim(0, 1)
        fig_in.savefig(os.path.join("save_figs", title))
        plt.close(fig_in)
        """
        
        #tf.print("phi_ini:",phi_ini.shape)
        #tf.print("X_lb.shape:",X_lb.shape)
        #tf.print("X_ub.shape:",X_ub.shape)
        #tf.print("X_ltb.shape:",X_ltb.shape)
        #tf.print("X_rtb.shape:",X_rtb.shape)
   
        #global list_loss_scipy
        #global Nfeval
        self.set_weights(parameters)

        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            
            loss_val, loss_BC,loss_IC, loss_f = self.loss(X_f,X_ini,X_lb,X_ub,X_ltb,X_rtb,phi_ini,self.abs_x_min,self.abs_x_max,self.abs_y_min,self.abs_y_max)   
            self.list_loss_scipy.append([loss_val, loss_BC,loss_IC, loss_f ])
            grads = tape.gradient(loss_val,self.trainable_variables)
                
        del X_ini,phi_ini,X_f, X_lb, X_ub, X_ltb, X_rtb

        grads_1d = [ ] #flatten grads 
        for i in range (len(self.layers)-1):
            grads_w_1d = tf.reshape(grads[2*i],[-1]) #flatten weights 
            grads_b_1d = tf.reshape(grads[2*i+1],[-1]) #flatten biases

            grads_1d = tf.concat([grads_1d, grads_w_1d], 0) #concat grad_weights 
            grads_1d = tf.concat([grads_1d, grads_b_1d], 0) #concat grad_biases
 
        #self.set_weights(grads_1d.numpy())
        del grads, grads_w_1d,grads_b_1d
        
        self.list_loss_scipy= self.optimizer_callback(parameters) # each pinn ==> list_loss_scipy 
                
        return loss_val.numpy(), grads_1d.numpy(), self.list_loss_scipy    
    ###############################################
    def optimizer_callback(self, parameters):  
             
        with self.Nfeval.get_lock():
            self.Nfeval.value += 1
            
            if self.Nfeval.value % 100 == 0:  # Print during scipy iterations 
                print_lock = multiprocessing.Lock()
                print_lock.acquire()
                tf.print("pinn: {:d}, Iter: {:d}, total_loss: {:.3e}, loss_BC: {:.3e}, loss_IC: {:.3e},loss_f: {:.3e}".format(self.idx_batch, self.Nfeval.value, \
                    self.list_loss_scipy[-1][0], self.list_loss_scipy[-1][1], self.list_loss_scipy[-1][2], self.list_loss_scipy[-1][3]))
                print_lock.release()

        return self.list_loss_scipy
    ##############################################
    def process_batch(self, batch_X_f, batch_X_ini, batch_X_lb, batch_X_ub,batch_X_ltb,batch_X_rtb, batch_phi_ini,model): #, 
        with tf.GradientTape() as tape:
            loss, loss_BC,loss_IC, loss_f = model.loss(batch_X_f,batch_X_ini,\
                                                                batch_X_lb,batch_X_ub,batch_X_ltb,batch_X_rtb,\
                                                                    batch_phi_ini,self.abs_x_min,self.abs_x_max,self.abs_y_min,self.abs_y_max)             
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer_Adam.apply_gradients(zip(gradients, model.trainable_variables))
            
        process_name = multiprocessing.current_process().name
        del tape, gradients
        #tf.print("Processor:", process_name)
        return loss, loss_BC, loss_IC, loss_f
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
        weights_files = self.PRE_POST.read_weights_files(path)
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
                                N_batches=N_batches,\
                                Nbr_f_pts_max_per_batch=self.Nbr_f_pts_max_per_batch,\
                                Nbr_f_pts_min_per_batch=self.Nbr_f_pts_min_per_batch,\
                                N_ini_max_per_batch=self.N_ini_max_per_batch,\
                                N_ini_min_per_batch=self.N_ini_min_per_batch)       

        for phase_idx in range(self.num_phases):
            phi_evolution_t_min = []
            phi_evolution_t_max = []
            X_phi_test_subsets = []
            n=len(PINN_.X_ini_train_all)
            N=PINN_.num_phases
            X_phi_test= PINN_.X_ini_train_all[int(n*phase_idx/N)+1:int(n*(phase_idx+1)/N)]
            phi_test= PINN_.phi_ini_train_all[int(n*phase_idx/N)+1:int(n*(phase_idx+1)/N)]
            
            # Predict evolution of each phase
            fig, ax = plt.subplots()
            
            for i in range(N_batches):
                if flag_Master==False:
                    pinn=self.pinns[i]
                else:
                    pinn=self # to use the same plot function for pinns and Master to save results 
                    
                t_min, t_max = self.t_min, self.t_max # self.PRE_POST.extract_t_min_t_max(weights_file)
                x_min, x_max, y_min, y_max = self.pinns[i].limits
                #tf.print("N_batches,x_min, x_max, y_min, y_max: ", N_batches,x_min, x_max, y_min, y_max)
                indices = np.where(
                (X_phi_test[:, 0] >= x_min) &
                (X_phi_test[:, 0] <= x_max) &
                (X_phi_test[:, 1] >= y_min) &
                (X_phi_test[:, 1] <= y_max)
                )

                X_phi_test_sub = X_phi_test[indices] 
                #flag_column = np.array([self.f_values[phase_idx]] * len(X_mat))[:, np.newaxis]
                #X_phi_test_sub= np.hstack((X_phi_test_sub,flag_column))
                phi_test_sub = phi_test[indices]
                if flag_t_max==False:        
                    X_phi_test_sub[:,2]=t_min   
                else:
                    X_phi_test_sub[:,2]=t_max   
                    #tf.print("here",X_phi_test_sub[:, :2].shape)
                phi_pred_t_min  = pinn.evaluate(X_phi_test_sub)
                phi=phi_pred_t_min
                #phi=np.clip(phi, 0, 1)
                scatter=ax.scatter(X_phi_test_sub[:, 0], X_phi_test_sub[:, 1], cmap=plt.get_cmap('viridis'), c=phi,vmin=0,vmax=1)
                
                x_avg = (x_min + x_max) / 2
                y_avg = (y_min + y_max) / 2

                ax.text(x_avg, y_avg, f"{i}", color='black', ha='right', va='bottom')
                
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
        
        # save_for_t_max
        #self.process_repository_files_discret_workers_Master(epoch,N_batches,\
        #path,pathOutput,path_weights_all_pinns,title,filename,flag_Master=False,flag_t_max=True)
        #del PINN_,phi_evolution,X_phi_test_sub,phi_pred
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
                                Nbr_f_pts_max_per_batch=self.Nbr_f_pts_max_per_batch,\
                                Nbr_f_pts_min_per_batch=self.Nbr_f_pts_min_per_batch,\
                                N_ini_max_per_batch=self.N_ini_max_per_batch,\
                                N_ini_min_per_batch=self.N_ini_min_per_batch)      
        
        phi_evolution = []
        num_train_intervals=self.num_train_intervals
        time_subdomains=np.linspace(0,1,num_train_intervals+1)
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
        title = f"φ predicted for epoch_{epoch+1} - t_min: {t_min:.3f}, t_max: {t_max:.3f}" # should be epoch+1 (this for debug purpose)
        filename = f"phi_pred_epoch_{epoch+1} - t_min: {t_min:.3f}, t_max: {t_max:.3f}.jpg"
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

        title = f"φ predicted for epoch_{epoch+1} - t_min: {t_min:.3f}, t_max: {t_max:.3f}" # should be epoch+1 (this for debug purpose)
        fig.suptitle(title)
      
        filename = f"phi_pred_epoch_{epoch+1} - t_min: {t_min:.3f}, t_max: {t_max:.3f}.jpg"
        plt.savefig(os.path.join(pathOutput ,filename))
        plt.close()
    ###############################################
    def save_weights(self, PINN,weights_file):
        weights_dict={}
        weights_key=0
        weights_dict[weights_key] = {}
        weights_dict[weights_key]['t_min'] = PINN.t_min
        weights_dict[weights_key]['t_max'] = PINN.t_max
        weights_loaded=tf.cast(PINN.get_weights(), dtype=tf.float64)
        weights_dict[weights_key]['weights'] = [w.numpy() for w in weights_loaded]
        with open(weights_file, 'w') as f:
            json.dump(weights_dict[weights_key], f)
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
    def save_weights_for_pinns(self, N_batches, path_weights_all_pinns, t_min, t_max, weights_key=0):
        
        # Erase of weights ( (N_batches+2)**2)
        self.PRE_POST.EraseFile(path=path_weights_all_pinns)
        for pinn_idx in range(len(self.pinns)):
            pinn =self.pinns[pinn_idx]
            pinn_path_weights = os.path.join(path_weights_all_pinns, f"pinn_{pinn_idx}")
            if not os.path.isdir(pinn_path_weights):
                os.mkdir(pinn_path_weights)
        
        tf.print("\n !!!  saving weights of pinns in progress !!! \n")
        num_processes = min(multiprocessing.cpu_count(), N_batches)  
        pool = multiprocessing.pool.ThreadPool(processes=num_processes)
        all_pinns_args_weights = []

        for pinn_idx in range(N_batches):
            pinn = self.pinns[pinn_idx]
            pinn_path_weights = os.path.join(os.path.join(path_weights_all_pinns, f"pinn_{pinn_idx}"),
                                            f"weights_tmin_{t_min:.5f}_tmax_{t_max:.5f}_{weights_key}.json")
            all_pinns_args_weights.append([pinn, pinn_path_weights])
            self.save_weights(pinn,pinn_path_weights)
        #pool.starmap(self.save_weights, all_pinns_args_weights)
        #pool.close()
        #pool.join()
        tf.print("\n     ! saving complete ! \n")
    ###############################################
    def set_weights_all_pinns(self,N_batches,pinns):
        tf.print("\n ! set initial weights for pinns! \n")
        path_get_weights_all_pinns=os.path.join(os.getcwd(),'get_weights_all_workers_pinns')
        path_weights_all_pinns=os.path.join(os.getcwd(),'weights_all_workers_pinns')
        weights_directories_in = [f for f in os.listdir(path_get_weights_all_pinns) if os.path.isdir(os.path.join(path_get_weights_all_pinns, f))]
        weights_directories_out = [f for f in os.listdir(path_weights_all_pinns) if os.path.isdir(os.path.join(path_weights_all_pinns, f))]

        if  len(weights_directories_in) == N_batches:
            for pinn_idx in range(N_batches):
                pinn = pinns[pinn_idx]
                pinn_path_weights =os.path.join(os.getcwd(),os.path.join(path_get_weights_all_pinns, f"pinn_{pinn_idx}") )
                weights_files = glob.glob(os.path.join(pinn_path_weights, '*.json'))
                weights_files = sorted(weights_files)
                weights_file = weights_files[0]
                
                match = re.search(r'_tmin_(\d+\.\d+)_tmax_(\d+\.\d+)_', weights_file)
                if match:
                    t_min = float(match.group(1))
                    t_max = float(match.group(2))
                    #print(t_min, t_max)
                else:
                    print("Unable to extract t_min and t_max.")
                    
                #print(pinn_idx,t_min, t_max)
                with open(weights_file, 'r') as f:
                    weights_loaded =json.load(f)['weights']
                weights_loaded=tf.cast(weights_loaded, dtype=self.precision)  
                pinn.set_weights(weights_loaded) 
                
        else:
        # If you have fewer weight directories than N_batches, subdivide the available directories
            tf.print("\n    ! Multiple to Multiple transfer learning ! \n")
            batch_size =  N_batches // len(weights_directories_in) 
            tf.print("\n batch_size: ", batch_size)
            for i, weights_dir in enumerate(weights_directories_in):
                weights_path = os.path.join(path_get_weights_all_pinns, weights_dir)
                weights_files = glob.glob(os.path.join(weights_path, '*.json'))
                weights_files = sorted(weights_files)

                for j in range(i * batch_size, min((i + 1) * batch_size, N_batches)):
                    #tf.print("here, i,j: ", i,j)
                    pinn = pinns[j]
                    pinn_path_weights =os.path.join(os.getcwd(),os.path.join(path_get_weights_all_pinns, f"pinn_{i}") )
                    weights_files = glob.glob(os.path.join(pinn_path_weights, '*.json'))
                    weights_files = sorted(weights_files)
                    weights_file = weights_files[0]
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
            #tf.print("index, limits: ",reset_pinn_idx,self.pinns[reset_pinn_idx].idx_batch, self.pinns[reset_pinn_idx].limits)
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
        Percentages_interface_points,path_weights_all_pinns,\
            coef_increase_points_f,coef_increase_points_ic,\
                Nbr_f_pts_max_per_batch,Nbr_f_pts_min_per_batch,\
                    N_ini_max_per_batch, N_ini_min_per_batch  ):
 
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
                sublimits_next_list.append([x_min, x_max, y_min, y_max])
   

        reduced_pinns_list=[] # reduced list from selected candidates
        reduced_pinns_reserve_list=[] # pinns from reduced list will be in the reserve list for the next training 
        list_indexes=[]
        list_new_weights=[]
        # list new weights 
        for index, (i, j) in enumerate([(i, j) for i in range(1, num_x_intervals-1) for j in range(1, num_y_intervals-1)]):
            index_ = i * num_y_intervals + j
            index_0=(i-1) * num_y_intervals + j-1
            index_1=i * num_y_intervals + j-1
            index_2=(i+1) * num_y_intervals + j-1
            index_3=(i-1) * num_y_intervals + j
            index_4=index_ 
            index_5=(i+1) * num_y_intervals + j
            index_6=(i-1) * num_y_intervals + j+1
            index_7= i* num_y_intervals + j+1
            index_8=(i+1) * num_y_intervals + j+1
            
            # First list 
            indexes=[index_4,index_0,index_1,index_2,index_3,index_5,index_6,index_7,index_8]
            percentages_interface_points = [Percentages_interface_points[idx] for idx in indexes]
            #for idx, percentage in zip(indexes, percentages_interface_points):
            #    #tf.print("new batch: ",index,"index:", idx, "Percentage:", percentage)
            max_index = indexes[percentages_interface_points.index(max(percentages_interface_points))]
            list_indexes.append(indexes)
  

            #reduced list : considering the spatial intersection of the small batch with the bin one 
            # ==>  same criterion as below + check the spatial intersection 
            reduced_indexes=[ ]  # for all batches containing interfacial points
            reduced_indexes_all=[ ] # for all batches containing interfacial and grain points (phi =0, phi =1, 0<phi<1)
            new_domain_limits=sublimits_next_list[index]  
            for idx_pinn in range(len(indexes)):
                pinn=self.pinns[indexes[idx_pinn]]
                if self.check_intersection(pinn, new_domain_limits)==True:
                    #tf.print("index: ", index, "idx_pinn: ",indexes[idx_pinn],Percentages_interface_points[indexes[idx_pinn]])
                    reduced_indexes_all.append(indexes[idx_pinn]) 
                    if  Percentages_interface_points[indexes[idx_pinn]]>0 :    
                        reduced_indexes.append(indexes[idx_pinn]) 
                        #tf.print("index: ",index,indexes[idx_pinn], Percentages_interface_points[indexes[idx_pinn]])
                    
            if len(reduced_indexes) > 0:
                    #tf.print("index: ",index, "reduced_indexes: ",reduced_indexes)
                    percentages_interface_points_ = [Percentages_interface_points[idx_] for idx_ in reduced_indexes]
                    max_index_ = reduced_indexes[percentages_interface_points_.index(max(percentages_interface_points_))]
                    #tf.print("len(reduced_indexes) > 0: ", "new batch: ",index,"==> selected pinn:", self.pinns[max_index_].idx_batch, " from the reduced_indexes:", reduced_indexes)
            else:
                    #for pinn in self.pinns_adam:
                    #    tf.print("pinn.idx_batch:", pinn.idx_batch)
                    target_index = index  # replace with the target index  to find the nearest f
                    idx_batch_values = [pinn.idx_batch for pinn in self.pinns_adam]
                    differences = [abs(idx - target_index) for idx in idx_batch_values]
                    nearest_index = differences.index(min(differences))
                    max_index_ = self.pinns_adam[nearest_index].idx_batch
            tf.print("New batch: ",index,"==> selected pinn:", self.pinns[max_index_].idx_batch)

            
             
            reduced_pinns_list.append(self.pinns[max_index_])
            reduced_pinns_reserve_list.append(reduced_indexes)
     
        # for the next training, each batch will have a reserve of weights to use them if the selected candidate can not success
        # ==> all candidate are queued through their arrays of weights which are kept (reserve_weights_per_pinn)
        all_pinns_reserve_weights=[ ]
        for idx_pinn in range(len(reduced_pinns_reserve_list)):
            reduced_indexes=reduced_pinns_reserve_list[idx_pinn]
            reserve_weights_per_pinn=[ ]
            for idx_ in range(len(reduced_indexes)) :
                reserve_weights_per_pinn.append(self.pinns[reduced_indexes[idx_]].get_weights() )
                #tf.print("self.pinns[idx_]:",  self.pinns[reduced_indexes[idx_]].idx_batch)    
            #tf.print("len(reserve_weights_per_pinn):",  len(reserve_weights_per_pinn), reduced_indexes)    
            all_pinns_reserve_weights.append(reserve_weights_per_pinn)
        
        ################################
        ######### Wichtig ##############
        ################################
        # here a new list of  pinns is selected
        # for next training 
        self.pinns=[]
        self.pinns=np.copy(reduced_pinns_list) #new_pinns_list 


        # get to each candidate a reserve list of wights, 
        # in other terms the four candidates are always present in the queue 
        # to participate in the training  if needed
        for reset_pinn_idx in range(len(self.pinns)):
            pinn = copy.copy(self.pinns[reset_pinn_idx])  # Create a shallow copy of pinn
            # before reset: self.pinns[reset_pinn_idx].idx_batch (idx of candidate in previous minibatching)
            #tf.print("before reset",self.pinns[reset_pinn_idx].idx_batch )
            pinn.idx_batch = reset_pinn_idx
            pinn.reseve_weights =[] # all_pinns_reserve_weights[reset_pinn_idx]  # uncomment for memory 
            self.pinns[reset_pinn_idx] = pinn
            #tf.print("after reset",self.pinns[reset_pinn_idx].idx_batch )
            #tf.print("HEEERRE  self.pinns[reset_pinn_idx]  ", len(self.pinns[reset_pinn_idx].reseve_weights ))
            # after reset: self.pinns[reset_pinn_idx].idx_batch (idx of candidate in next minibatching)
            #tf.print("HEEERRE",reset_pinn_idx, len(self.pinns), self.pinns[reset_pinn_idx].idx_batch,self.pinns[reset_pinn_idx].limits )  
        self.set_new_limits()
        
        #for pinn_idx in range(len(self.pinns)):
        #    tf.print(self.pinns[pinn_idx].idx_batch, self.pinns[pinn_idx].limits)
        
        ################################
        ################################
        ################################
        """
        # averaging the weights: thid option to test 
        all_weights=[]
        for index in indexes:
            weights = self.pinns[index].get_weights()
            all_weights.append(weights)
        avg_weights = [np.mean([weights[k] for weights in all_weights]) for k in range(len(all_weights))]
        self.pinns[max_index].set_weights(avg_weights)
        tf.print("setting weights in progress")
        new_pinns_list.append(self.pinns[max_index])
        """

        
         
        # update sizes training batches 
        Nbr_f_pts_max_per_batch=int(Nbr_f_pts_max_per_batch *coef_increase_points_f)
        Nbr_f_pts_min_per_batch= int(Nbr_f_pts_min_per_batch*coef_increase_points_f)
        N_ini_max_per_batch=int(N_ini_max_per_batch*coef_increase_points_ic)
        N_ini_min_per_batch=int(N_ini_min_per_batch*coef_increase_points_ic )     
        tf.print("Nbr_f_pts_max_per_batch:", Nbr_f_pts_max_per_batch)
        tf.print("Nbr_f_pts_min_per_batch:", Nbr_f_pts_min_per_batch)
        tf.print("N_ini_max_per_batch:", N_ini_max_per_batch)
        tf.print("N_ini_min_per_batch:", N_ini_min_per_batch)
        return  Nbr_f_pts_max_per_batch,Nbr_f_pts_min_per_batch,N_ini_max_per_batch,N_ini_min_per_batch
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
        plt.title(f'X_ini_train (n={batch_X_ini.shape[0]})\nFor the training interval: t_min: {t_min:.3f}, t_max: {t_max:.3f}')
        filename = f"IC_epoch_{epoch+1} - For the training interval: t_min: {t_min:.3f}, t_max: {t_max:.3f}.png"
        plt.savefig(os.path.join(path, filename))
        plt.close()
    ##################################################################################################
    ##################################################################################################
    ####################################       Train       ###########################################
    ##################################################################################################
    ##################################################################################################
    def train(self,epochs,batch_size_max,thresh,epoch_scipy_opt=1000,epoch_print=500, epoch_resample=100,\
            initial_check=False,save_reg_int=100,num_train_intervals=10,\
            discrete_resolv=True,fraction_ones_per_int_pts=0.3,coef_increase_points_f=2,coef_increase_points_ic=2,\
            path=None,pinns=None, path_weights_all_pinns=None,save_weights_pinns=True,\
                communicate_pinns=False,change_pinn_candidate=False,Thresh_Master=0.1,optimize_master=False): 
         
        # time intervals 
        time_subdomains=np.linspace(0,1,num_train_intervals+1)
        count=0
        #thresh
        self.thresh=thresh
        # init
        X_ini=self.X_ini  # N_ini points (user selection)
        phi_ini=self.phi_ini  
        X_ini_all=self.X_ini_train_all # All ini points (to use by the PINN if mini_batches are not filled)
        phi_ini_all=self.phi_ini_train_all # All ini points (to use by the PINN if mini_batches are not filled)
        
        N_batches=self.N_batches

        # loss
        list_loss_workers=[]

        #self.pinns=pinns
        
        # dummy params (flags)
        flag=0
        flag_weights=1
        flag_shuffle_for_scipy=0
        flag_over_fit=0
        alpha=1
        flag_reload=0
        flag_reduce_batches=0
        flag_train_Master=0
        flag_print_master=1
        flag_print_ignore=1
        
        do_master_loop=0
        flag_start_Master=0
        count_=0
        debug_scipy_master=1
        flag_scipy_optimize_pinns= 0 
        count_scipy_iter=0
        dummy_flag=1
        
        # shuffle Collocation points
        idx = np.random.permutation(self.X_f.shape[0])
        self.X_f = self.X_f[idx]
  
        # get N_b and N_ini
        N_b=0#self.X_lb.shape[0]
        N_ini=self.X_ini.shape[0]

        # Create a dictionary to store the weights for each time interval 
        weights_dict = {}
        weights_key = 0
        # Get current process ID
        pid = os.getpid()
        # Get current process object
        process = psutil.Process(pid)
        with open("usage_log.txt", "w") as f_mem:
            ############################################
            ############### EPOCH LOOP #################
            ############################################
            if N_batches >= 1:  # security step
                for epoch in range(epochs):  
                    ####################
                    if epoch==0 or flag_reduce_batches==1 or flag :
                        if flag_reduce_batches==1: # N_batches already reduced ==> take the correspondig pinn selected candidates
                            pinns=self.pinns 
                            count_=0
                        ####################
                        if epoch==0 or flag :  # a new time interval ==> restart (initial number of points, batches, pinns)
                            Nbr_f_pts_max_per_batch=np.copy(self.Nbr_f_pts_max_per_batch)
                            Nbr_f_pts_min_per_batch=np.copy(self.Nbr_f_pts_min_per_batch)
                            N_ini_max_per_batch= np.copy(self.N_ini_max_per_batch)
                            N_ini_min_per_batch=np.copy(self.N_ini_min_per_batch)
                            N_batches=int(np.copy(self.N_batches))
                            if epoch>0:
                                pinns=self.re_Initialize_pinns()  

                            #if epoch>0: 
                            #    #tf.print("check: ", flag,flag_train_Master, flag_train_workers ,flag_reduce_batches,N_batches )
                        tf.print("\n -------------------------------------------------------------")
                        tf.print("  -----  Epoch: {0:d} <==> N_batches: {1:d}, pinns: {2:d}    -------".format(epoch,N_batches,len(pinns)))
                        if epoch==0:
                            tf.print("  ----- time domain: ",'t_min: {0:.5f}, t_max: {1:.5f}'.format(self.t_min, 1/self.Nt))
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
                    if epoch>=10:
                        self.f=1
                        self.ic=1
                        #self.bc=1  
                        total_time=self.ub[2]-self.lb[2] 
                        actual_time=t_max-self.lb[2]
                        self.bc=0#((total_time-actual_time)/total_time)*1e-3
                        if flag_train_workers==1:
                            self.lr=0.001
                        else:
                            self.lr=0.001 

                    if flag_reload==1:
                        # Search for the weight file with matching t_min and t_max values
                        weights_dir = 'get_weights/'
                        filename_pattern = f'weights_tmin_{t_min:.3f}_tmax_{t_max:.3f}_*.json'
                        matching_files = glob.glob(weights_dir + filename_pattern)
                        if matching_files:
                            weights_file = sorted(matching_files)[-1]
                            tf.print("t_min, t_max: ",t_min, t_max)
                            tf.print("load weights from: ",weights_file)
                            
                            with open(weights_file, 'r') as f:
                                weights_loaded =json.load(f)['weights']
                            weights_loaded=tf.cast(weights_loaded, dtype=self.precision)
                            self.set_weights(weights_loaded)                    
                    
                    if epoch % epoch_resample==0:
                        # re-shuffle X_f (Collocation) training points
                        idx = np.random.permutation(self.X_f.shape[0])
                        self.X_f = self.X_f[idx]

                    #if epoch==0 or flag:
                    # update selection from Collocation points
                    X_f_sub_domain = self.X_f[np.logical_and(t_min <= self.X_f[:,2], self.X_f[:,2] <= t_max)]

                    # update selection from bouńdaries)
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
                    
                    # **************** X_f for scipy  ***** 
                    # Collocation points to use by scipy optimizer (it is the same as used by Adam but contructed in a different way)
                    
                    if flag_over_fit==1:
                        tf.print("flag_over_fit=1")
                        flag_shuffle_for_scipy=1
                                            
                    if epoch==0 or flag==1 or flag_shuffle_for_scipy==1 or flag_reduce_batches==1 or flag_start_Master==1:
                        X_f_sub_domain_scipy = []
                        #*************************************                    
                        # **************** IC ****************
                        #*************************************
                        # Select the corresponding initial points based on the selected indices
                        X_ini_all_sub_domain = X_ini_all#[selected_indices_xy]
                        phi_ini_all_sub_domain = phi_ini#_all[selected_indices_xy]

                        self.X_ini_all_sub_domain=X_ini_all_sub_domain
                        self.phi_ini_all_sub_domain = phi_ini_all_sub_domain

                        # move the IC points
                        if epoch==0 or flag==1 and flag_shuffle_for_scipy==0:
                            if flag_train_Master==0:
                                flag_train_workers=1 # training workers-based until convergence and reaching next time interval

                            if discrete_resolv and epoch>0:
                                    X_ini_all_sub_domain[:,2]=t_min
                                    phi_ini_all_sub_domain = self.evaluate(X_ini_all_sub_domain).numpy()
    
                                    self.X_ini_all_sub_domain=X_ini_all_sub_domain
                                    self.phi_ini_all_sub_domain=phi_ini_all_sub_domain
                                    a= stop_here
                                    #Debug 
                                    #plt.scatter(self.X_ini_all_sub_domain[:, 0], self.X_ini_all_sub_domain[:, 1], cmap=plt.get_cmap('viridis'), c=np.sum(self.phi_ini_all_sub_domain, axis=1, keepdims=True),alpha=0.8)
                                    #plt.colorbar( shrink=0.35)
                                    #plt.show()
                        # *********************************************************************
                        # --------------   prepare scipy IC points  ---------------------------
                        # ---  Note this is note only for scipy but IC poitns for adam too-----
                        # *********************************************************************
                        selected_indices_scipy = []  # IC indices for scipy optimizer
                        #tf.print(self.phi_ini_all_sub_domain.shape,self.phi_ini_all_sub_domain[200:400])
                        # Iterate over the intervals and randomly select indices within each interval
                        row_sums = np.sum(self.phi_ini_all_sub_domain, axis=1)
                        interfacial_indices = np.where(np.logical_and(row_sums > 0, row_sums < 1))[0]
                        interfacial_indices_sampled = np.random.choice(interfacial_indices, size=int(len(interfacial_indices) * 1), replace=False)
                        zero_indices = np.where(row_sums == 0)[0]
                        one_indices = np.where(row_sums == 1)[0]
                        """ 
                        one_indices = []
                        for i in range(len(self.phi_ini_all_sub_domain)):
                            row = self.phi_ini_all_sub_domain[i]
                            if row==1:
                                one_indices.append(i)
                            "
                            row_sum = np.sum(row)
                            if row_sum==1:
                                for j in range(len(row)):
                                    if row[j]==1: 
                                        one_indices.append(i)
                                        break 
                                else:
                                    break
                                   
                        one_indices = np.unique(one_indices) 
                        """ 
                                              
                        # Select % of the int indices and one indices
                        if len(zero_indices>0):
                            zero_indices_sampled = np.random.choice(zero_indices, size=int(len(zero_indices) * 0.1), replace=False)
                        if len(one_indices>0):
                            one_indices_sampled = np.random.choice(one_indices, size=int(len(interfacial_indices_sampled) * fraction_ones_per_int_pts), replace=False)
                        all_indices_sampled = np.concatenate([interfacial_indices_sampled, one_indices_sampled,zero_indices_sampled])
                        #tf.print("all_indices_sampled", len(zero_indices_sampled),len(one_indices_sampled),len(interfacial_indices_sampled))
                    
                        #**************************
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
                                batch_indices = np.where((X_ini_all_sub_domain_reduced[:, 0] >= x_lb) & (X_ini_all_sub_domain_reduced[:, 0] < x_ub) &
                                                            (X_ini_all_sub_domain_reduced[:, 1] >= y_lb) & (X_ini_all_sub_domain_reduced[:, 1] < y_ub))[0]
                                                            
                                sum_phi_columns = np.sum(self.phi_ini_all_sub_domain[all_indices_sampled], axis=1)
                                
                                int_phi_values=sum_phi_columns[batch_indices]
        
                                int_thresh=0.05 # a small filter to not to consider noisy areas 
                                total_interface_points = np.sum((int_phi_values > 0 + int_thresh) & (int_phi_values < 1 - int_thresh))
                                
                                #tf.print("total_interface_points: ",i * num_y_intervals + j,total_interface_points,len(batch_indices) )
                                if len(batch_indices)>0:
                                    percentage_interface_points = total_interface_points / len(np.asarray(self.phi_ini_all_sub_domain[all_indices_sampled])[batch_indices])
                                else:
                                    percentage_interface_points=0
                                                            
                                Percentages_interface_points.append(percentage_interface_points)
                                #tf.print("total_interface_points: ", i * num_y_intervals + j,percentage_interface_points,N_ini_max_per_batch,N_ini_max_per_batch)
                                N_ini_per_batch = min(int(percentage_interface_points *N_ini_max_per_batch), N_ini_max_per_batch)

                                if len(batch_indices) > 0:   # number of points per batch
                                    #tf.print("size: ",len(batch_indices),max(N_ini_per_batch,N_ini_min_per_batch))
                                    if len(batch_indices)<max(N_ini_per_batch,N_ini_min_per_batch):
                                        replace=True  
                                    else:
                                        replace =False 
                                        #tf.print("\n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n")
                                        #tf.print("\n !!! Check your IC Training batch !!! \n")
                                        #tf.print("len(batch_indices) =", len(batch_indices), ", max(N_ini_per_batch, N_ini_min_per_batch) =", int(max(N_ini_per_batch, N_ini_min_per_batch)))
                                        #tf.print("\n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n")
                                    list_ini_per_batch=np.random.choice(batch_indices, size=max(N_ini_per_batch,N_ini_min_per_batch), replace=replace)
                                    selected_indices_scipy.extend(list_ini_per_batch)
                                    Total_list_indices_ini.append(list_ini_per_batch)
                                    #tf.print("size: ",max(N_ini_per_batch,N_ini_min_per_batch))
                                    
                        self.indices_ini =selected_indices_scipy # update indices for scipy optimization
                                                                # for each subdomain self.indices will return the IC points  
                        
                        X_ini_all_sub_domain_reduced=np.asarray(X_ini_all_sub_domain_reduced)
                        phi_ini_all_sub_domain_reduced=np.asarray(phi_ini_all_sub_domain_reduced)
                        
                        X_ini_sub_domain = X_ini_all_sub_domain_reduced[selected_indices_scipy]
                        phi_ini_sub_domain = phi_ini_all_sub_domain_reduced[selected_indices_scipy] 

                        self.X_ini_sub_domain=X_ini_all_sub_domain_reduced[self.indices_ini]
                        self.phi_ini_sub_domain=phi_ini_all_sub_domain_reduced[self.indices_ini]                    
                        
                        # numerotaion of pinns (to_def )
                        for i in range(N_batches):
                            x_min, x_max, y_min, y_max=sublimits_list[i]
                            x_avg = (x_min + x_max) / 2
                            y_avg = (y_min + y_max) / 2
                            plt.text(x_avg, y_avg, f"{i}", color='black', ha='right', va='bottom',fontsize=5)
                
                        plt.scatter(X_ini_sub_domain[:, 0], X_ini_sub_domain[:, 1],s=0.5,\
                            cmap=plt.get_cmap('jet'), c=np.sum(phi_ini_sub_domain,axis=1))
                        plt.colorbar( shrink=0.35)
                        title=f"Scipy optimizer IC points at Epoch {epoch} for Time interval: t_min: {t_min:.3f}, t_max: {t_max:.3f}.jpg"
                        phi_ini_length = len(np.asarray(self.phi_ini_all_sub_domain)[self.indices_ini])
                        plt.title(f'Number of IC points for Scipy optimization: {phi_ini_length}',fontsize=8)
                        plt.grid(True)
                        plt.xticks(np.linspace(0, 1, num_x_intervals+1))
                        plt.yticks(np.linspace(0, 1, num_y_intervals+1))
                        if flag_shuffle_for_scipy==0:
                            plt.savefig(os.path.join(path,title), dpi = 500, bbox_inches='tight')
                        plt.close()            
        
                        # *********************************************
                        # check initial condition (epoch==0 or domain change)
                        #if initial_check:
                        #    self.plot_ini(X_ini_sub_domain,phi_ini_sub_domain,X_ini,self.Nx,self.Ny,path,t_min,t_max, epoch)
                                
                        """
                        # for debug 
                        # for training 
                        X_ini_train_c=X_ini_sub_domain
                        phi_ini_train_c=phi_ini_sub_domain
                        plt.scatter(X_ini_train_c[:, 0], X_ini_train_c[:, 1], cmap=plt.get_cmap('viridis'), c=np.sum(phi_ini_train_c, axis=1, keepdims=True),alpha=0.8)
                        plt.colorbar( shrink=0.35)
                        plt.show()

                        # reduced ini all (without zeros)
                        X_ini_train_all_reduc=X_ini_all_sub_domain_reduced
                        phi_ini_train_all_reduc=phi_ini_all_sub_domain_reduced
                        plt.scatter(X_ini_train_all_reduc[:, 0], X_ini_train_all_reduc[:, 1], cmap=plt.get_cmap('viridis'), c=np.sum(phi_ini_train_all_reduc, axis=1, keepdims=True),alpha=0.8)
                        plt.colorbar( shrink=0.35)
                        plt.show()
                        
                        # all ini subdomain 
                        X_ini_train_all_c=self.X_ini_all_sub_domain[int(40804*3/4)+1:int(40804*4/4)]
                        phi_ini_train_all_c=self.phi_ini_all_sub_domain[int(40804*3/4)+1:int(40804*4/4)]
                        plt.scatter(X_ini_train_all_c[:, 0], X_ini_train_all_c[:, 1], cmap=plt.get_cmap('viridis'), c=np.sum(phi_ini_train_all_c, axis=1, keepdims=True),alpha=0.8)
                        plt.colorbar( shrink=0.35)
                        plt.show()
                        """
                    ####################################################################################################
                    ####################################################################################################
                    ###############  Minibatching   #################################################################### 
                    ####################################################################################################
                    ####################################################################################################
                    # Define the number of minibatches : to correct
                    #roat_N_batches = batch_size
                    flag_filled= True
                    # Define the number of intervals for x and y
                    #num_x_intervals = int(np.ceil(np.sqrt(roat_N_batches)))
                    #num_y_intervals = int(np.ceil(np.sqrt(roat_N_batches)))
                    #N_batches = int(num_x_intervals*num_y_intervals)

                    # Calculate the interval size for x and y
                    #x_interval_size = (X_lb_sub_domain[:, 0].max() - X_lb_sub_domain[:, 0].min()) / num_x_intervals
                    #y_interval_size = (X_ltb_sub_domain[:, 1].max() - X_ltb_sub_domain[:, 1].min()) / num_y_intervals

                    # Create a list of subdomains
                    all_batches_Xf = []
                    all_batches_Xf_for_pred  = []
                    all_batches_limits = []
                    all_batches_X_ini =[] 
                    all_batches_phi_ini = []
                    all_batches_X_lb = []
                    all_batches_X_ub = []
                    all_batches_X_ltb = []
                    all_batches_X_rtb = []
                    all_batches_X_ini_all =[] # All ini points (to use by the PINN if mini_batches are not filled)
                    all_batches_phi_ini_all = [] # All ini points (to use by the PINN if mini_batches are not filled)
                    
                    # Loop over the intervals and group the points based on their location  
                    if epoch==0 or flag or flag_reduce_batches==1:     
                        fig, ax = plt.subplots()          
                    for i in range(num_x_intervals):
                        for j in range(num_y_intervals):
                            index_batch = i * num_y_intervals + j
                            
                            # Define the boundaries for the subdomain
                            x_min = X_lb_sub_domain[:, 0].min() + i * x_interval_size
                            x_max = X_ub_sub_domain[:, 0].min() + (i + 1) * x_interval_size
                            y_min = X_ltb_sub_domain[:, 1].min() + j * y_interval_size
                            y_max = X_rtb_sub_domain[:, 1].min() + (j + 1) * y_interval_size
                            
                            # Select the points in X_f_sub_domain that fall within the X_f_batch boundaries
                            batch_Xf_indices = np.where(
                                (X_f_sub_domain[:, 0] >= x_min) &
                                (X_f_sub_domain[:, 0] <= x_max) &
                                (X_f_sub_domain[:, 1] >= y_min) &
                                (X_f_sub_domain[:, 1] <= y_max)
                            )[0].astype(int)
                            batch_Xf = X_f_sub_domain[batch_Xf_indices]

                            # Select the points in X_ini that fall within the subdomain boundaries
                            bacth_X_ini_indices = np.where(
                                (X_ini_sub_domain[:, 0] >= x_min) &
                                (X_ini_sub_domain[:, 0] <= x_max) &
                                (X_ini_sub_domain[:, 1] >= y_min) &
                                (X_ini_sub_domain[:, 1] <= y_max)
                            )[0].astype(int)
                            batch_X_ini = X_ini_sub_domain[bacth_X_ini_indices]
                            if (len(batch_X_ini)==0):
                                flag_filled=False
                            batch_phi_ini = phi_ini_sub_domain[bacth_X_ini_indices]

                            # Select the points in X_ini_all that fall within the subdomain boundaries
                            bacth_X_ini_all_indices = np.where(
                                (X_ini_all_sub_domain_reduced[:, 0] >= x_min) &
                                (X_ini_all_sub_domain_reduced[:, 0] <= x_max) &
                                (X_ini_all_sub_domain_reduced[:, 1] >= y_min) &
                                (X_ini_all_sub_domain_reduced[:, 1] <=
                                y_max)
                            )[0].astype(int)
                            bacth_X_ini_all_indices=np.asarray(bacth_X_ini_all_indices)
                            
                            batch_X_ini_all = X_ini_all_sub_domain_reduced[bacth_X_ini_all_indices]
                            batch_phi_ini_all = phi_ini_all_sub_domain_reduced[bacth_X_ini_all_indices]
                            # *********************************************************************
                            # ******************     Adaptive minibatching   Scipy  ***************
                            # *********************************************************************                  
                            percentage_interface_points_per_batch = Percentages_interface_points[index_batch]
                            
                            number_f_points_per_batch=max(Nbr_f_pts_min_per_batch,int(percentage_interface_points_per_batch*Nbr_f_pts_max_per_batch))
                            
                            if len(batch_Xf)<number_f_points_per_batch:
                                replace_=True  
                            else:
                                replace_=False

                            if len(bacth_X_ini_all_indices) > 0:
                                new_indices_collocation_pts=np.random.choice(len(batch_Xf), size=number_f_points_per_batch, replace=replace_)
                                batch_Xf_for_pred=np.copy(batch_Xf)  # all subdomain for prediction
                                batch_Xf=batch_Xf[new_indices_collocation_pts] # for training 
                            
                            ### new 
                            batch_X_lb_indices = np.where(
                                (X_lb_sub_domain[:, 0] >= x_min) &
                                (X_lb_sub_domain[:, 0] <= x_max) 
                                )[0].astype(int)
                            batch_X_lb = X_lb_sub_domain[batch_X_lb_indices]

                            batch_X_ub_indices = np.where(
                                (X_ub_sub_domain[:, 0] >= x_min) &
                                (X_ub_sub_domain[:, 0] <= x_max) 
                                )[0].astype(int)
                            batch_X_ub = X_ub_sub_domain[batch_X_ub_indices]

                            batch_X_ltb_indices = np.where(
                                (X_ltb_sub_domain[:, 1] >= y_min) &
                                (X_ltb_sub_domain[:, 1] <= y_max) 
                                )[0].astype(int)
                            batch_X_ltb = X_ltb_sub_domain[batch_X_ltb_indices]                 

                            batch_X_rtb_indices = np.where(
                                (X_rtb_sub_domain[:, 1] >= y_min) &
                                (X_rtb_sub_domain[:, 1] <= y_max) 
                                )[0].astype(int)
                            batch_X_rtb = X_rtb_sub_domain[batch_X_rtb_indices]

                            all_batches_Xf.append(batch_Xf)
                            all_batches_Xf_for_pred.append(batch_Xf_for_pred)
                            all_batches_limits.append([x_min,x_max,y_min,y_max])
                            all_batches_X_ini.append(batch_X_ini)
                            all_batches_phi_ini.append(batch_phi_ini)
                            all_batches_X_lb.append(batch_X_lb)
                            all_batches_X_ub.append(batch_X_ub)
                            all_batches_X_ltb.append(batch_X_ltb)
                            all_batches_X_rtb.append(batch_X_rtb)
                            all_batches_X_ini_all.append(batch_X_ini_all)
                            all_batches_phi_ini_all.append(batch_phi_ini_all)
                                                
                            # plot the subdomain
                            if epoch==0 or flag or flag_reduce_batches==1:    
                                color = np.random.rand(3)
                                ax.scatter(batch_Xf[:, 0], batch_Xf[:, 1], color=color, marker='*',s=0.5, label='PDE Collocation')
                                ax.scatter(batch_X_ini[:, 0], batch_X_ini[:, 1], color=color, marker='o',s=25, label='IC')
                                ax.scatter(batch_X_lb[:, 0], batch_X_lb[:, 1], color=color, marker='v',s=5, label='Lower Boundary')
                                ax.scatter(batch_X_ub[:, 0], batch_X_ub[:, 1], color=color, marker='^',s=5, label='Upper Boundary')
                                ax.scatter(batch_X_ltb[:, 0], batch_X_ltb[:, 1], color=color, marker='<',s=5, label='Left Boundary')
                                ax.scatter(batch_X_rtb[:, 0], batch_X_rtb[:, 1], color=color, marker='>',s=5, label='Right Boundary')
                                x_avg = (x_min + x_max) / 2
                                y_avg = (y_min + y_max) / 2
                                ax.text(x_avg, y_avg, f"{index_batch}", color='orange', ha='right', va='bottom')
                                
                    all_batches_Xf = np.asarray(all_batches_Xf, dtype=object)       
                    all_batches_X_lb = np.asarray(all_batches_X_lb, dtype=object)    
                    all_batches_X_ub = np.asarray(all_batches_X_ub, dtype=object)
                    all_batches_X_ltb = np.asarray(all_batches_X_ltb, dtype=object)    
                    all_batches_X_rtb = np.asarray(all_batches_X_rtb, dtype=object)           
                    all_batches_X_ini=np.asarray(all_batches_X_ini, dtype=object)        
                    all_batches_phi_ini=np.asarray(all_batches_phi_ini, dtype=object)

                    # Concatenate all the tensors 
                    all_batches_X_ini_to_plot = all_batches_X_ini[0]
                    all_batches_phi_ini_to_plot = all_batches_phi_ini[0]
                                
                    for i in range(1, len(all_batches_X_ini)):
                        #tf.print("Here 1",all_batches_X_ini[i].shape,all_batches_phi_ini[i].shape)
                        all_batches_X_ini_to_plot = np.vstack((all_batches_X_ini_to_plot, all_batches_X_ini[i]))
                        #phi_ini_reshaped = np.reshape(all_batches_phi_ini[i], (1, -1))
                        all_batches_phi_ini_to_plot = np.vstack((all_batches_phi_ini_to_plot, all_batches_phi_ini[i]))
                                
                    if epoch==0 or flag==1 or flag_reduce_batches==1:  
                        phases_value= np.reshape(np.sum(all_batches_phi_ini_to_plot,axis=1), (-1, 1))
                        scatter_ini=ax.scatter(all_batches_X_ini_to_plot[:, 0], all_batches_X_ini_to_plot[:, 1],cmap=plt.get_cmap('viridis') ,c=phases_value,marker='o',s=10, label="IC")  
                        cbar = plt.colorbar(scatter_ini, ax=ax, shrink=0.35, label=r"$\phi$")
                        title = f"Adam Subdomains\nEpoch {epoch}, {int(num_x_intervals*num_x_intervals)} minibatches\n(t_min = {t_min:.3f}, t_max = {t_max:.3f})" 
                        plt.title(title, fontsize=10)
                        if flag_filled:
                            plt.text(0.5, 0.5, r"$\bf{IC\ points\ well\ filled}$", color="orange", fontsize=16, ha="center", va="center")
                        else: 
                            plt.text(0.5, 0.5, "!!! IC points auto-filled",color="red" , fontsize=14, ha="center", va="center")
                        #plt.legend()
                        plt.savefig(os.path.join(path, f"Adam_Subdomains_Epoch_{epoch}_{int(num_x_intervals*num_x_intervals)}_minibatches_t_min_{t_min:.3f}_t_max_{t_max:.3f}.jpg"), dpi=500, bbox_inches='tight')
                        plt.close() 
                        ####################################################################

                    ######################################################################################
                    ######################################################################################
                    ######################################################################################
                    # Main loop for minibatching : compute loss for each batch  ##########################
                    ######################################################################################
                    ######################################################################################
                    ######################################################################################
                    batch_args = []  # for parallelization
                    pinns_adam = []
                    if N_batches==self.min_batch_numbers:
                        batch_args_Master=[]   # Data for Training Master with Adam 
                        batch_args_scipy_Master=[]  # Data for optimizing Master with Adam 
                    #tf.print("self.All_interfaces_ini",tf.reduce_sum(self.All_interfaces_ini, axis=0))
                    random_indices = random.sample(range(N_batches), 3) # for check 
                    for batch_idx, (batch_X_f, batch_X_ini,batch_X_ini_all,batch_phi_ini,batch_phi_ini_all,\
                                    batch_X_lb, batch_X_ub, batch_X_ltb, batch_X_rtb)\
                                        in enumerate(zip(all_batches_Xf, all_batches_X_ini,all_batches_X_ini_all,\
                                                            all_batches_phi_ini,all_batches_phi_ini_all, all_batches_X_lb,\
                                                                all_batches_X_ub, all_batches_X_ltb, all_batches_X_rtb)):
                        pinns[batch_idx].flag=0                    
                                            
                        if len(batch_X_ini)==0:  # fill this batch with IC points and corresponding phi (to reactivate later if interface taken into account)
                            idx_ini_all = np.random.choice(batch_X_ini_all.shape[0], N_ini_min_per_batch, replace=True)
                            #tf.print("idx_ini_all: ",len(idx_ini_all))
                            batch_X_ini=batch_X_ini_all[idx_ini_all]
                            batch_phi_ini=batch_phi_ini_all[idx_ini_all]
                                            
                        # update batch list for self.pinns for scipy                
                        self.X_ini_sub_domain = tf.concat([self.X_ini_sub_domain, batch_X_ini], axis=0)
                        self.phi_ini_sub_domain = tf.concat([self.phi_ini_sub_domain, batch_phi_ini], axis=0)
                        
                        if Percentages_interface_points[batch_idx]>0.1:
                                pinns[batch_idx].flag=1
                        pinns[batch_idx].idx_batch=batch_idx
                        
                        if pinns[batch_idx].flag==0: # internal code management,
                            sampled_indices = np.random.choice(len(batch_X_ini), 5, replace=False)
                            batch_X_ini = batch_X_ini[sampled_indices]
                            batch_phi_ini = batch_phi_ini[sampled_indices]
                            sampled_indices_ = np.random.choice(len(batch_X_f), 5, replace=False)
                            batch_X_f= batch_X_f[sampled_indices_]
                        
                        if (epoch==0 or flag==1  or flag_reduce_batches==1 ) and (batch_idx in random_indices and  pinns[batch_idx].flag>0) and flag_shuffle_for_scipy==0:
                            self.PRE_POST.plot_domain(X_ini_sub_domain,batch_X_ini,X_f_sub_domain,batch_X_f,X_ub_sub_domain,batch_X_ub,\
                                                        X_lb_sub_domain,batch_X_lb,X_ltb_sub_domain,batch_X_ltb,X_rtb_sub_domain,batch_X_rtb,\
                                                        t_min, t_max,epoch,batch_idx,phi_0=tf.reduce_sum(self.All_interfaces_ini, axis=0) ,phi_ini_train=np.sum(phi_ini_sub_domain,axis=1),phi_ini_train_s=np.sum(batch_phi_ini,axis=1),path=path)  
                                                    
                        batch_X_f = tf.convert_to_tensor(batch_X_f, dtype=self.precision)
                        batch_X_ini = tf.convert_to_tensor(batch_X_ini, dtype=self.precision)
                        batch_X_lb = tf.convert_to_tensor(batch_X_lb, dtype=self.precision)
                        batch_X_ub = tf.convert_to_tensor(batch_X_ub, dtype=self.precision)
                        batch_X_ltb = tf.convert_to_tensor(batch_X_ltb, dtype=self.precision)
                        batch_X_rtb = tf.convert_to_tensor(batch_X_rtb, dtype=self.precision)
                        batch_phi_ini = tf.convert_to_tensor(batch_phi_ini, dtype=self.precision)
                        batch_arg=[batch_X_f, batch_X_ini, batch_X_lb, batch_X_ub,batch_X_ltb,batch_X_rtb, batch_phi_ini,pinns[batch_idx]] 
                        
                        if pinns[batch_idx].flag>0:
                                if count_< 10 : #to avoid brutal transition
                                    pinns_adam.append(pinns[batch_idx])
                                    batch_args.append(batch_arg)  
                                    if N_batches==self.min_batch_numbers:
                                        batch_args_Master.append([batch_X_f, batch_X_ini, batch_X_lb, batch_X_ub,batch_X_ltb,batch_X_rtb, batch_phi_ini] )
                                else:
                                    #tf.print( "pinns[batch_idx]: ", pinns[batch_idx].idx_batch, "Epoch: ", epoch, "count :", count_) 
                                    if (pinns[batch_idx].idx_batch not in self.pinns_adam_below_thresh) and (pinns[batch_idx].idx_batch in self.pinns_adam_above_thresh) and (pinns[batch_idx].idx_batch not in self.pinns_adam):
                                        #tf.print(" pinns[batch_idx].idx_batch: ",pinns[batch_idx].idx_batch, " Epoch: ", epoch, "count :", count_)
                                        pinns_adam.append(pinns[batch_idx])
                                        batch_args.append(batch_arg)
                                    if N_batches==self.min_batch_numbers:
                                        batch_args_Master.append([batch_X_f, batch_X_ini, batch_X_lb, batch_X_ub,batch_X_ltb,batch_X_rtb, batch_phi_ini] )

                        pinn_data_for_scipy=[batch_X_f, batch_X_ini_all, batch_X_lb, batch_X_ub,batch_X_ltb,batch_X_rtb, batch_phi_ini_all]
                        
                        if N_batches==self.min_batch_numbers:
                            batch_args_scipy_Master.append(pinn_data_for_scipy)
                            
                        pinns[batch_idx].pinn_data_for_scipy= pinn_data_for_scipy  # Update the 'self.batch' attribute with 'batch_args'
                        pinns[batch_idx].batch_Xf_for_pred=all_batches_Xf_for_pred[batch_idx]
                        pinns[batch_idx].limits=all_batches_limits[batch_idx]

                        pinns[batch_idx].Nfeval =multiprocessing.Value('i', 0)   # reset (for later scipy optimization)
                        pinns[batch_idx].lr=self.lr   # always update pinns learning rate 
                        self.pinns=pinns 
                        #tf.print("Here:", len(self.pinns),batch_idx,self.pinns[batch_idx].limits  )
                        """
                        # check debug (limits, collocation and IC of each batch etc.)
                        data=pinns[batch_idx].batch_Xf_for_pred 
                        tf.print("check: ",data[:,0].min(), data[:,0].max(),data[:,1].min(), data[:,1].max(),data[:,2].min(), data[:,2].max()  )
                        plt.scatter(data[:,0],data[:,1])
                        plt.show()
                        tf.print("check: ","batch", batch_idx, pinns[batch_idx].limits)
                        for i in range(len(pinn_data_for_scipy)):
                            tf.print("pinns[batch_idx].batch_arg shape:", np.asarray(pinns[batch_idx].pinn_data_for_scipy[i]).shape)              
                        """
                            
                        # Update Scipy Collocation points list during minibatching    
                        if (epoch==0 or flag==1 or flag_shuffle_for_scipy==1 or flag_reduce_batches==1 or flag_start_Master==1) :     
                            for batch_Xf_row in zip(batch_X_f):                           
                                X_f_sub_domain_scipy.append(batch_Xf_row)
                    count_ +=1                          
                    ################################################
                    ################################################
                    ############    Parallell  #####################
                    ############## Training   ######################
                    ################################################
                    ################################################
                    if len(pinns_adam)>0:
                        self.pinns_adam=pinns_adam
                        
                    if flag_train_workers==1 and len(batch_args) >0:                          
                        #tf.print("Epoch: ", epoch,"len(batch_args): ",len(batch_args) )
                        num_processes=min(multiprocessing.cpu_count(),len(batch_args) )# a maximum of  processes are available

                        pool = multiprocessing.pool.ThreadPool(processes=num_processes)
                        results = pool.starmap(self.process_batch, batch_args)
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

                        for batch_idx, (loss, loss_BC, loss_IC, loss_f) in enumerate(results):                            
                            pinn =copy.copy(pinns_adam[batch_idx])
                            #pinns[pinn.idx_batch]=pinn 
                            #if flag_reduce_batches:
                            #  tf.print("len(batch_args): ",len(batch_args))
                            #  tf.print("pinn: ",pinn.idx_batch)
                            #  tf.print("len(pinns): ",len(pinns))
                            
                            global_loss_workers += loss
                            global_loss_f_workers += loss_f
                            global_loss_IC_workers += loss_IC
                            global_loss_BC_workers += loss_BC
            
                            if loss < min_loss:
                                min_loss = loss
                                min_loss_idx = pinn.idx_batch
                            if loss > max_loss:
                                max_loss = loss
                                max_loss_idx = pinn.idx_batch

                                    
                            if loss < self.thresh and pinn.idx_batch not in pinns_adam_below_thresh and  pinn.flag==1:  
                                pinns_adam_below_thresh.append(pinn.idx_batch)  
                                if flag_scipy_optimize_pinns== 1 and pinn.idx_batch in pinns_above_avg and  pinn.flag==1:    # consider if now optimized      
                                    pinns_above_avg.remove(pinn.idx_batch) 
                                    tf.print("removed pinn: ", pinn.idx_batch, ", from pinns_above_avg ") 
                                    if pinn.idx_batch not in pinns_below_avg: # security if (it should be)
                                        pinns_below_avg.append(pinn.idx_batch)                        
                                
                            if loss >= self.thresh and pinn.idx_batch not in pinns_adam_above_thresh and  pinn.flag==1: 
                                pinns_adam_above_thresh.append(pinn.idx_batch) 
                                # to continue scipy optimization 
                                if flag_scipy_optimize_pinns== 1 and pinn.idx_batch not in pinns_above_avg and  pinn.flag==1:   #not consider once already optimized by scipy previous iter
                                    pinns_adam_above_thresh.remove(pinn.idx_batch)    # to not be considered by scipy (already optimized by scipy, but during return to adam the loss could be slightly increased)                          max_loss=self.thresh-1e-3  # particular case, to artifialy pass the loss below the thresh once pinn.idx_batch is not considered
                                    
                        del pinn, results
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
                        self.pinns_adam_below_thresh = [pinn.idx_batch for pinn in self.pinns if pinn.idx_batch not in idx_batch_above_thresh]                          
                    
                    ########################################################################
                    ##############################      END      ###########################
                    ####################     Parrallel training  of pins ###################
                    ########################################################################
                    ########################################################################                       
                        list_loss_workers.append([global_loss_workers,global_loss_BC_workers,global_loss_IC_workers,global_loss_f_workers])                    

                    elif flag_train_Master==1:

                        if  flag_print_master==1 :
                            tf.print("\n ---------------------------------- \n")
                            tf.print("\n ---- Training the Master PINN ---- \n")
                            tf.print("\n ---------------------------------- \n")
                            tf.print("N_batches= ",N_batches)
                            tf.print(" !!! Initiate training  ==>  batch: ", max_index_Master)
                            count_iter = 0
                            indices_to_use = [max_index_Master] 
                            other_indexes = [0,1,2,3]
                            other_indexes.remove(max_index_Master) 
                            flag_print_master=0
                            
                            
                        global_loss_Master = 0
                        global_loss_f_Master= 0
                        global_loss_IC_Master = 0
                        global_loss_BC_Master= 0
                                                                             
                        for batch_idx in range(N_batches):
                             if batch_idx in indices_to_use:
                                batch_X_f=batch_args_Master[batch_idx][0]
                                batch_X_ini=batch_args_Master[batch_idx][1]
                                batch_X_lb=batch_args_Master[batch_idx][2]
                                batch_X_ub=batch_args_Master[batch_idx][3]
                                batch_X_ltb=batch_args_Master[batch_idx][4]
                                batch_X_rtb=batch_args_Master[batch_idx][5]
                                batch_phi_ini=batch_args_Master[batch_idx][-1]

                                if len(batch_X_ini)==0:  # fill this batch with IC points and corresponding phi 
                                    tf.print("Autofill, batch:", batch_idx)
                                    idx_ini_all = np.random.choice(batch_X_ini_all.shape[0], N_ini_min_per_batch, replace=False)
                                    batch_X_ini=batch_X_ini_all[idx_ini_all]
                                    batch_phi_ini=batch_phi_ini_all[idx_ini_all]
                                    batch_phi_ini = batch_phi_ini.reshape(-1, 1)
                                    self.X_ini_sub_domain = tf.concat([self.X_ini_sub_domain, batch_X_ini], axis=0)
                                    self.phi_ini_sub_domain = tf.concat([self.phi_ini_sub_domain, batch_phi_ini], axis=0)
                                
                                if flag_start_Master==1 and batch_idx==0: # raise flag_shuffle_for_scipy for debug
                                    self.PRE_POST.plot_domain(X_ini_sub_domain,batch_X_ini,X_f_sub_domain,batch_X_f,X_ub_sub_domain,batch_X_ub,\
                                                                                            X_lb_sub_domain,batch_X_lb,X_ltb_sub_domain,batch_X_ltb,X_rtb_sub_domain,batch_X_rtb,\
                                                                                            t_min, t_max,epoch,batch_idx,phi_0=tf.reduce_sum(self.All_interfaces_ini, axis=0) ,phi_ini_train=np.sum(phi_ini_sub_domain,axis=1),phi_ini_train_s=np.sum(batch_phi_ini,axis=1),path=path) 
                                    
                                batch_X_f = tf.convert_to_tensor(batch_X_f, dtype=self.precision)
                                batch_X_ini = tf.convert_to_tensor(batch_X_ini, dtype=self.precision)
                                batch_X_lb = tf.convert_to_tensor(batch_X_lb, dtype=self.precision)
                                batch_X_ub = tf.convert_to_tensor(batch_X_ub, dtype=self.precision)
                                batch_X_ltb = tf.convert_to_tensor(batch_X_ltb, dtype=self.precision)
                                batch_X_rtb = tf.convert_to_tensor(batch_X_rtb, dtype=self.precision)
                                batch_phi_ini = tf.convert_to_tensor(batch_phi_ini, dtype=self.precision)
                                
                                with tf.GradientTape() as tape:
                                    loss, loss_BC,loss_IC, loss_f = Master_PINN.loss(batch_X_f,batch_X_ini,\
                                                                            batch_X_lb,batch_X_ub,batch_X_ltb,batch_X_rtb,\
                                                                                batch_phi_ini,self.abs_x_min,self.abs_x_max,self.abs_y_min,self.abs_y_max)
                                    gradients = tape.gradient(loss, Master_PINN.trainable_variables)
                                    results=Master_PINN.optimizer_Adam.apply_gradients(zip(gradients, Master_PINN.trainable_variables))
                                
                                global_loss_Master += loss
                                global_loss_f_Master += loss_f
                                global_loss_IC_Master += loss_IC
                                global_loss_BC_Master += loss_BC
                                
                        global_loss_Master /= len(indices_to_use)
                        global_loss_f_Master /= len(indices_to_use)
                        global_loss_IC_Master /= len(indices_to_use)
                        global_loss_BC_Master/= len(indices_to_use)
                    
                        del tape,gradients
                        count_iter+=1
                    
                        if len(other_indexes) > 0:
                            if count_iter % 50 == 0 and global_loss_Master < Thresh_Master:
                                tf.print("other_indexes: ", other_indexes)
                                tf.print("indices_to_use: ", indices_to_use)
                                differences = [abs(idx - max_index_Master) for idx in other_indexes]
                                tf.print("differences: ", differences)
                                #nearest_index = differences.index(min(differences))
                                nearest_index = other_indexes[np.argmin(differences)]
                                tf.print("nearest_index: ", nearest_index)
                                if nearest_index not in indices_to_use:
                                    indices_to_use.append(nearest_index)
                                if nearest_index in other_indexes:
                                    other_indexes.remove(nearest_index)
                                else:
                                    print(f"The value {nearest_index} is not in other_indexes.")
                                tf.print("!!! Expand training domain ==> add batch: ", nearest_index)

                 
                        do_master_loop=1 # to be sure to enter this loop 
                        flag_start_Master=0
                    ########################################################################
                    ##############################      END      ###########################
                    ########################      MASTER training ########################
                    ########################################################################
                    ########################################################################
                    if  epoch==0 or flag==1 or flag_reduce_batches==1 or flag_start_Master==1 and flag_shuffle_for_scipy==0:      
                        # Get and check Scipy Collocation points list         
                        X_f_sub_domain_scipy=np.asarray(X_f_sub_domain_scipy)
                        X_f_sub_domain_scipy = np.reshape(X_f_sub_domain_scipy, (len(X_f_sub_domain_scipy), 4))
                        self.X_f_sub_domain_scipy=X_f_sub_domain_scipy # update 
                        
                        # plot scipy points for check
                        plt.scatter(self.X_f_sub_domain_scipy[:, 0], self.X_f_sub_domain_scipy[:, 1],s=0.1, cmap=plt.get_cmap('jet'), c=self.X_f_sub_domain_scipy[:, 2])
                        cbar =plt.colorbar( shrink=0.5)                        
                        cbar.set_label("Time")

                        title=f"Scipy optimizer Collocation points at Epoch {epoch} for the Time interval: t_min: {t_min:.3f}, t_max: {t_max:.3f}.jpg"
                        len_X_f_sub_domain_scipy = len(self.X_f_sub_domain_scipy)
                        plt.title(f"Scipy optimizer Collocation points at Epoch {epoch} \n for the Time interval: t_min: {t_min:.3f}, t_max: {t_max:.3f}: {len_X_f_sub_domain_scipy :.0f} points",fontsize=10)
                        plt.grid(True)
                        plt.xlabel('x')
                        plt.ylabel('y')
                        plt.xticks(np.linspace(X_lb_sub_domain[:, 0].min(), X_ub_sub_domain[:, 0].max(), num_x_intervals+1))
                        plt.yticks(np.linspace(X_ltb_sub_domain[:, 1].min(),X_ltb_sub_domain[:, 1].max(),num_y_intervals+1))
                        plt.savefig(os.path.join(path,title), dpi = 500, bbox_inches='tight')
                        plt.close()      
                        
                        flag=0  # very important : all data are now prepared for minibatching / training ==> drop the flag 
                        flag_shuffle_for_scipy=0
                        flag_reduce_batches=0
                        
                        
                        del X_f_sub_domain_scipy
                        
                    # Delete variables to free up memory
                    del all_batches_Xf, all_batches_X_ini, all_batches_phi_ini, all_batches_X_lb, all_batches_X_ub, all_batches_X_ltb, \
                        all_batches_X_rtb, all_batches_X_ini_all, all_batches_phi_ini_all,batch_X_f,batch_X_ini,\
                                                                    batch_X_lb,batch_X_ub,batch_X_ltb,batch_X_rtb,\
                                                                        batch_phi_ini, new_indices_collocation_pts                                              
                        
                    ####################################################################################################
                    ####################################################################################################
                    ###############  End Minibatching   ################################################################
                    ####################################################################################################
                    ####################################################################################################              
                    # Save predictions at regular time intervals
                    if (epoch+1)  % save_reg_int== 0:  
                        self.save_predictions_regular_int(epoch,path,self.X_phi_test,\
                                    X_ini,phi_ini,N_b,t_min, t_max)

                    # print losses
                    if epoch % epoch_print == 0 and flag_train_workers==1 and flag_train_Master==0:    
                        tf.print('\n==> Epoch: {0:d}, Mean_loss of pinns: {1:.3e}, loss_BC: {2:.3e}, loss_IC: {3:.3e}, loss_f: {4:.3e}'.format(epoch, global_loss_workers, global_loss_BC_workers,global_loss_IC_workers, global_loss_f_workers))
                        tf.print(' => minimum loss: {0:.3e}, corresponding pinn/batch index: {1:d}'.format(min_loss, min_loss_idx))
                        tf.print(' => maximum loss: {0:.3e}, corresponding pinn/batch  index: {1:d}'.format(max_loss, max_loss_idx))
                        if count_-1>= 10 and (len(batch_args)>0):
                            #tf.print("count_", count_-1)
                            tf.print(" => In : Number of pinns to train: {0:d}".format(len(batch_args))) 
                            tf.print(" => Out : Number of pinns above Threshold: {0:d}".format(len(self.pinns_adam_above_thresh))) 
                            #tf.print(" => Out: Number of pinns below Threshold: {0:d}".format(len(pinns_adam_below_thresh)))
                            tf.print("pinns above Threshold:", ", ".join(map(str, self.pinns_adam_above_thresh)))

                            #tf.print("Pinns below Threshold:", [idx_batch for idx_batch in pinns_adam_below_thresh])
                            #tf.print("Threshold : {0:.3e}\n".format(self.thresh))
                    
                    if epoch % epoch_print == 0 and flag_train_Master==1 and flag_train_workers==0: 
                        tf.print('\n Epoch: {0:d}, total_loss: {1:.3e}, loss_BC: {2:.3e}, loss_IC: {3:.3e}, loss_f: {4:.3e} \n'.format(epoch, global_loss_Master, global_loss_BC_Master,global_loss_IC_Master, global_loss_f_Master))
                    
                    ##################################################
                    ##################################################
                    ###############  Scipy Optimizer pinns  ##########
                    ##################################################
                    ##################################################   
                    #if self.alpha>1 and  epoch % epoch_print == 0:
                    #    tf.print("\n !!!!!! After Adam Optimization : !!!!!!")
                    #    tf.print(" !!! number of pinns above the Treshhold: ", len(pinns_above_avg))
                    #    tf.print(" !!! pinns above the Treshhold (self.thresh) ==>", ", ".join(map(str, pinns_above_avg)))
                    ##################################################
                    ##################################################                     
                    # call scipy optimizer if loss > thresh
                    if epoch % epoch_scipy_opt == 0 and max_loss > self.thresh and epoch>0 and flag_train_workers==1 and len(pinns_adam)>0 : #and N_batches==self.N_batches: 
                        tf.print("\n !!! Scipy optimize: !!! - Epoch: ",str(epoch))
                        #global Nfeval
                        Nfeval=1  # reinitialize the global value of Nfeval
                        global list_loss_scipy
 
                        batch_args_scipy= []
                        reduced_list_pinns=[]
                        for pinn_idx in range(N_batches):
                            pinn=pinns[pinn_idx]
                            pinn.list_loss_scipy = []
                            if flag_scipy_optimize_pinns== 0:
                                if pinn.flag==1 and pinn.idx_batch in pinns_adam_above_thresh:
                                    batch_args_scipy.append([pinn,pinn.get_weights().numpy()])
                                    reduced_list_pinns.append(copy.copy(pinn))
                                
                            elif flag_scipy_optimize_pinns== 1:  # self.alpha>1 ==>reduced list to optimize weights (those whose loss still above thresh)
                                if pinn.idx_batch in pinns_above_avg and pinn.idx_batch in pinns_adam_above_thresh and  pinn.flag==1: 
                                    reduced_list_pinns.append(copy.copy(pinn))
                                    pinn=pinns[pinn_idx]
                                    batch_args_scipy.append([pinn,pinn.get_weights().numpy()])
                        processes=len(batch_args_scipy)

                        # to change simply to pinns_to_optimize=reduced_list_pinns 
                        if self.alpha==1: 
                            pinns_to_optimize=reduced_list_pinns #copy.copy(pinns)   # optimize all pinns 
                        else:
                            pinns_to_optimize= reduced_list_pinns # optimize only pinns whose loss>max_loss
                        
                        # Parallel optimization
                        num_processes_scipy = min(16, processes) # maximum number of cores for parallelization  (max of 16 by security)
                        tf.print("Processes: ", num_processes_scipy)
                        tf.print("Number of pinns to optimize weights: ", len(pinns_to_optimize))
                        pinns_to_optimize_indexes = [str(pinn.idx_batch) for pinn in pinns_to_optimize]
                        tf.print(" !!! pinns to optimize ==>", ", ".join(pinns_to_optimize_indexes))
                        tf.print("\n")
                        pool_scipy = multiprocessing.pool.ThreadPool(processes=num_processes_scipy)
                        results = pool_scipy.starmap(self.optimize_single_pinn, batch_args_scipy)
                        pool_scipy.close()
                        pool_scipy.join()
                        
                        del reduced_list_pinns, batch_args_scipy
                        
                        # debug 
                        #self.save_predictions_discret_workers_Master(epoch,path,path_weights_all_pinns,self.X_phi_test,\
                        #X_ini,phi_ini,N_b,t_min, t_max,N_batches)
                        
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
                            pinn_idx=pinn.idx_batch
                            #best_result = min(result, key=lambda x: x.fun)
                            pinn.set_weights(result.x)
                            pinns[pinn_idx].set_weights(result.x)
                            loss,loss_BC,loss_IC,loss_f=  pinn.list_loss_scipy[-1][0], pinn.list_loss_scipy[-1][1], pinn.list_loss_scipy[-1][2], pinn.list_loss_scipy[-1][3]
                            list_of_losses.append(loss)
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
                        #tf.print("here, len(list_of_losses) :  ",len(list_of_losses)) #N_batches)    
                        global_loss_workers /= len(list_of_losses) #N_batches
                        global_loss_f_workers /= len(list_of_losses) #N_batches
                        global_loss_IC_workers /= len(list_of_losses) #N_batches
                        global_loss_BC_workers/= len(list_of_losses) #N_batches
                        
                        pinns_below_avg = []
                        pinns_above_avg = []
                        for idx, loss in enumerate(list_of_losses):
                            pinn_idx=pinns_to_optimize[idx].idx_batch
                            pinn = pinns[pinn_idx]
                            num_interface_points = Percentages_interface_points[pinn_idx]  
                        
                            if loss < self.thresh:
                                pinns_below_avg.append(pinns[pinn_idx].idx_batch)
                            else:
                                pinns_above_avg.append(pinns[pinn_idx].idx_batch)  

                        tf.print("\n !!! number of pinns above the Treshhold: ", len(pinns_above_avg))
                        tf.print("\n !!! pinns above the Treshhold ==>", ", ".join(map(str, pinns_above_avg)))
                        tf.print("\n !!! Scipy optimization done !!!\n ")
                        flag_scipy_optimize_pinns=1
                        
                        tf.print('==> loss after L-BFGS-B optimization for Epoch: {0:d}, Mean_loss of pinns: {1:.3e}, loss_BC: {2:.3e}, loss_IC: {3:.3e}, loss_f: {4:.3e}'.format(epoch, global_loss_workers,global_loss_BC_workers,global_loss_IC_workers, global_loss_f_workers))
                        tf.print(' => minimum loss: {0:.3e}, corresponding pinn/batch index: {1:d}'.format(min_loss, min_loss_idx))
                        tf.print(' => maximum loss: {0:.3e}, corresponding pinn/batch  index: {1:d}'.format(max_loss, max_loss_idx))
                        
                        if max_loss > self.thresh: 
                                self.alpha,Nbr_f_pts_max_per_batch,\
                                     Nbr_f_pts_min_per_batch=self.increase_points(Nbr_f_pts_max_per_batch, Nbr_f_pts_min_per_batch)            
                                tf.print('\n Scipy - reshuffle at epoch {}, for a total loss: {:.3e}\n'.format(epoch,global_loss_workers))
                                if self.alpha > 1 : 
                                    flag_shuffle_for_scipy=1
                                    count_scipy_iter+=1
                                    Nbr_f_pts_max_per_batch,Nbr_f_pts_min_per_batch=self.revert_the_increase(Nbr_f_pts_max_per_batch, Nbr_f_pts_min_per_batch)  # added for check purpose, could be retained
                                    """
                                    tf.print('\n Epoch {}, self.alpha: {}, Nbr_f_pts_max_per_batch: {},  Nbr_f_pts_min_per_batch: {}\n'.format(
                                            epoch,
                                            int(self.alpha),
                                            int(Nbr_f_pts_max_per_batch),
                                            int(Nbr_f_pts_min_per_batch)
                                        ))
                                    """
                                    
                                if self.alpha > 10:  # reduce computing time      # to delete is check purpose ..... ( if condition above)
                                    Nbr_f_pts_max_per_batch,Nbr_f_pts_min_per_batch=self.revert_the_increase (Nbr_f_pts_max_per_batch, Nbr_f_pts_min_per_batch) 
                                    tf.print('\n Epoch {}, self.alpha {}, Nbr_f_pts_max_per_batch: {},  Nbr_f_pts_min_per_batch: {}\n'.format(
                                            epoch,
                                            int(self.alpha),
                                            int(Nbr_f_pts_max_per_batch),
                                            int(Nbr_f_pts_min_per_batch)
                                        ))
                                    
                                idx_batch_above_avg = [idx_batch for idx_batch in pinns_above_avg]
                                list_below_thresh = [pinn.idx_batch for pinn in self.pinns if pinn.idx_batch not in idx_batch_above_avg]

                                if count_scipy_iter>10 and len(pinns_above_avg)>0 and len(list_below_thresh) > 0 and dummy_flag==1:  #  get weights from another pinn 
                                    tf.print(" pinns_below_avg:", list_below_thresh)
                                    tf.print(" number of scipy iterations:", count_scipy_iter)
                                    for idx_pinn in pinns_above_avg:
                                        differences = [abs(idx_pinn - idx) for idx in list_below_thresh]
                                        nearest_index = list_below_thresh[differences.index(min(differences))]
                                        
                                        tf.print("==> get for pinn: ",idx_pinn," weights of the  pinn:", nearest_index)
                                        pinns[idx_pinn].set_weights(pinns[nearest_index].get_weights())
                                        count_scipy_iter=0 
                                        dummy_flag=0
                                    
                                    
                                if communicate_pinns==True:
                                    for pinn_idx in pinns_above_avg:
                                        #tf.print("communicate pinns: setting new weights to pinns with losses above average")
                                        rand_idx = random.sample(range(len(pinns_below_avg)), 1)
                                        pinns[pinn_idx].set_weights(pinns[rand_idx[0]].get_weights())
                                if change_pinn_candidate==True:
                                    for pinn_idx in pinns_above_avg:
                                        pinn_reserve_weights=pinns[pinn_idx].reseve_weights
                                        if len(pinn_reserve_weights)>0:
                                                rand_idx = random.sample(range(len(pinn_reserve_weights)), 1)
                                                pinns[pinn_idx].set_weights(pinn_reserve_weights[rand_idx[0]])
                                                tf.print('\n weights changed for pinn: ', pinns[pinn_idx].idx_batch)
                                        else:
                                                tf.print('\n empty list of reserve weights for pinn: ', pinns[pinn_idx].idx_batch)
                                    
                        #if max_loss > 1e-1:   # value by trial (the loss should be too small)
                        #    flag_over_fit=1
                        #    alpha=alpha+1  # to add later (scipy optimization)
                        
                        #else:
                        #    if max_loss < 1e-2: # value by trial (generally no overfit if < 1e-2)
                        #            flag_over_fit=0
                        #            alpha=alpha+1      
                                 
                        if flag_over_fit==1:
                                self.lr/=2 
                                minimum_learning_rate=5e-4
                                if self.lr>=minimum_learning_rate:
                                    tf.print("\n !!! over-fit ==> decrease the learning rate of pinns to {:.2e}\n ".format(self.lr)) 
                                else:
                                    self.lr=minimum_learning_rate

                        del results,result
              
                    ##################################################
                    ##################################################
                    ###############  Scipy Optimizer  Master  ########
                    ##################################################
                    ##################################################  
                    
                    # call scipy optimizer if loss > thresh
                    if do_master_loop==1 and optimize_master==True: 
                        if epoch % epoch_scipy_opt == 0 and global_loss_Master > Thresh_Master and flag_train_workers==0 and flag_train_Master==1  and epoch>0: 
                            global list_loss_scipy
                            list_loss_scipy = []
                            init_params = Master_PINN.get_weights().numpy()
                            tf.print("\n")
                            tf.print("!!! Scipy optimize: !!! - Epoch: ",str(epoch))

                            Master_PINN.Nfeval_master=multiprocessing.Value('i', 1)  # reinitialize the global value of Nfeval
                            
                            results = scipy.optimize.minimize(fun = Master_PINN.optimizerfunc, 
                                                            x0 = init_params, 
                                                            args=(), 
                                                            method= 'L-BFGS-B', 
                                                            jac= True,        # If jac is True, fun is assumed to return the gradient along with the objective function
                                                            callback = Master_PINN.optimizer_callback_master, 
                                                            options = {'disp': None,
                                                                        'maxiter': 1000,    # f_ 
                                                                        'iprint': -1})
                            Master_PINN.set_weights(results.x)
                            Master_PINN.lr=0.0001
                            global_loss_Master,global_loss_BC,global_loss_IC, global_loss_f =list_loss_scipy[-1]
                            global_loss_Master,global_loss_BC_Master,global_loss_IC_Master, global_loss_f_Master =list_loss_scipy[-1]
        
                            #list_loss.append([loss,loss_BC, loss_IC,loss_f]) # to reactivate
                            tf.print('==> loss after L-BFGS-B optimization for Epoch: {0:d}, total_loss: {1:.3e}, loss_BC: {2:.3e}, loss_IC: {3:.3e}, loss_f: {4:.3e}'.format(epoch, global_loss_Master,global_loss_BC_Master,global_loss_IC_Master, global_loss_f_Master))
                            tf.print("!!! Scipy optimization done !!!\n ")
                            if global_loss_Master > Thresh_Master: 
                                    flag_shuffle_for_scipy=1                     
                                    tf.print('Scipy - reshuffle, epoch, total loss:', epoch, "{:.3e}".format(global_loss_Master))
                                        
                            del results,list_loss_scipy,init_params
                            

                            #debug 
                            if debug_scipy_master==1:
                                self.set_weights(Master_PINN.get_weights())
                                self.save_predictions_discret_workers_Master(epoch,path,path_weights_all_pinns,self.X_phi_test,\
                                X_ini,phi_ini,N_b,t_min, t_max,N_batches,flag_Master=True) 
                                debug_scipy_master=0
                    ########################################################################
                    ########################################################################
                    
                    if flag_train_workers==1: 
                        thresh_loss=np.copy(max_loss)
                    elif flag_train_Master==1 and flag_train_workers==0 : 
                        thresh_loss=np.copy(global_loss_Master)
                    if thresh_loss < self.thresh and  epoch % epoch_print == 0 and flag_train_workers==0 :
                        Nbr_f_pts_max_per_batch,Nbr_f_pts_min_per_batch=self.revert_the_increase(Nbr_f_pts_max_per_batch, Nbr_f_pts_min_per_batch)
                        #tf.print("Nbr_f_pts_max_per_batch,Nbr_f_pts_min_per_batch, alpha  ",Nbr_f_pts_max_per_batch,Nbr_f_pts_min_per_batch,self.alpha) 
                        
                    if flag_scipy_optimize_pinns== 1:   #not consider once already optimized by scipy previous iter
                        if len(pinns_adam_above_thresh)==0 and flag_print_ignore==1:   
                            tf.print("\n ! ignoring pinn(s) whose loss < Threshold and change domain")                   
                            thresh_loss=self.thresh-1e-4 # particular case, to artifialy pass the loss below the thresh once pinn.idx_batch is already not considered    
                            # explication: case after scipy optmization all pinns losses are below the thresh, a return to some adam iterations
                            # is mandatory to avoid brutal changes of domains (from 36 batches to directly 4 for e.g. )
                            # the loss a given(s) pinn(s) could slightly increase ; we should tolerate this and allow this and allow the change    
                            flag_print_ignore=0
                    ########################################################################
                    ########################################################################
                    ###############  Save and change Domain (time or N:batches ) ###########
                    ########################################################################
                    ########################################################################                                           
                    if thresh_loss < self.thresh and t_max<=1 and flag==0 and epoch>0 and epoch % 17==0 : #the last is added to ensure that the model train a bit before new changes and continuity of the minbatching 
                        if flag_train_workers==1: 
                            tf.print("\n max_loss: {0:.3e} < Threshold: {1:.3e}\n".format(thresh_loss, self.thresh))
                            #tf.print(" =>  Number of pinns above Threshold: {0:d}\n".format(N_batches-len(pinns_adam_below_thresh)))
                            count_=0 # reset 
                            count_scipy_iter=0 # reset 
                            flag_print_ignore=1 # reset
                        ################################################################
                        if  flag_weights:  # save weights at each time-domain change 
                            Nbr_f_pts_max_per_batch,Nbr_f_pts_min_per_batch=self.revert_the_increase(Nbr_f_pts_max_per_batch, Nbr_f_pts_min_per_batch)  
                            # save predictions 
                            ################################
                            if discrete_resolv: 
                                if flag_train_workers==1:  # workers pinns 
                                    self.save_predictions_discret_workers_Master(epoch,path,path_weights_all_pinns,self.X_phi_test,\
                                            X_ini,phi_ini,N_b,t_min, t_max,N_batches,flag_Master=False)
                                    if N_batches==self.min_batch_numbers:
                                        self.save_weights_for_pinns(N_batches, path_weights_all_pinns, t_min, t_max)
                                    # # will be activated again in the next time interval
                                    flag_reduce_batches=1  # reduce batches  
                                    flag_scipy_optimize_pinns= 0 # re-ini
                                ################################      
                                if flag_reduce_batches==1:
                                    #self.thresh/=1
                                    #tf.print("new self.thresh: " ,self.thresh)
                                    # reduce number of batches 
                                    N_batches=int((np.sqrt(N_batches)-2)**2)
                                    if N_batches>1:
                                        tf.print('==> Epoch: {0:d}, Mean_loss of pinns: {1:.3e}, loss_BC: {2:.3e},\
                                            loss_IC: {3:.3e}, loss_f: {4:.3e}'.format(epoch, global_loss_workers,global_loss_BC_workers,global_loss_IC_workers, global_loss_f_workers))
                                        tf.print("\n reduce N_batches from Epoch: {0:d} ==>  N_batches: {1:d}\n".format(epoch+1,N_batches))
                                        tf.print("\n max_loss: {0:.3e}, min_loss: {1:.3e}\n".format(max_loss, min_loss))
                                        # update the list of pinns basing on a Pyramid-like or a Convolutional- like selection    
                                        Nbr_f_pts_max_per_batch,Nbr_f_pts_min_per_batch,N_ini_max_per_batch,N_ini_min_per_batch=self.reduce_batches_Pyramid_like_training(N_batches,num_x_intervals,num_y_intervals,\
                                            Percentages_interface_points,path_weights_all_pinns,coef_increase_points_f,coef_increase_points_ic,\
                                                            Nbr_f_pts_max_per_batch,Nbr_f_pts_min_per_batch,\
                                                            N_ini_max_per_batch, N_ini_min_per_batch )                                     
                                    
                                    elif N_batches < self.min_batch_numbers:  
                                        flag_train_workers=0
                                        flag_reduce_batches=0 
                                        N_batches=self.min_batch_numbers
                                        flag_train_Master=1 
                                        do_master_loop=0
                                        count_scipy_iter=0 # reset 
                                        
                                        self.pinns=pinns 
                                        percentages_interface_points = [Percentages_interface_points[idx] for idx in range(len(Percentages_interface_points))]
                                        max_index_Master = percentages_interface_points.index(max(percentages_interface_points))
                                        tf.print("max_idx ", max_index_Master, len(self.pinns) )
                                        for pinn_idx in range(len(self.pinns)):
                                            tf.print("pinn.idx_batch ",pinn_idx, self.pinns[pinn_idx].idx_batch)
                                        
                                        # Warm starting of Master PINN
                                        Master_PINN=self.pinns[max_index_Master]
                                        Master_PINN.N_batches=self.N_batches
                                        Master_PINN.pinns=np.copy(pinns)
                                        Master_PINN.set_new_limits_Master()
                                        self.set_weights(Master_PINN.get_weights())
                                        Master_PINN.lr=0.0001
                                        flag_start_Master=1
                                        
                                        
                                        #self.save_predictions_discret_workers_Master(epoch,path,path_weights_all_pinns,self.X_phi_test,\
                                        #        X_ini,phi_ini,N_b,t_min, t_max,N_batches,flag_Master=True) 
                                        tf.print("\n warm starting of the Master PINN ===> selected candidate: pinn ", str(int(max_index_Master)))
                                        self.thresh=Thresh_Master
                                        tf.print("\n ")
                            ################################    
                            else: # continuous resolv 
                                self.save_predictions_continous(epoch,path,self.X_phi_test,\
                                    X_ini,phi_ini,N_b,t_min, t_max)

                            if t_max==1:   # stop saving weigthts (alles abgeschlossen)
                                self.thresh=global_loss_Master/1.1
                                tf.print("Now optimizing the solution for the new threshold: {:.3e}".format(self.thresh))
                                #flag_weights=0                            

                        # Save Master PREDICTIONS and prepare training on the next time domain
                        if t_max<1 and N_batches==self.min_batch_numbers and flag_train_Master==1 and epoch % 17==0 and do_master_loop==1 and global_loss_Master<Thresh_Master and len(other_indexes)==0:
                            tf.print("\n N_batches:", N_batches)
                            tf.print("flag_train_Master:", flag_train_Master)
                            tf.print("epoch :", epoch)
                            tf.print("do_master_loop:", do_master_loop)
                            tf.print("global_loss_Master: {0:.3e}".format(global_loss_Master))
                            
                            self.save_predictions_discret_workers_Master(epoch,path,path_weights_all_pinns,self.X_phi_test,\
                            X_ini,phi_ini,N_b,t_min, t_max,N_batches,flag_Master=True) 
                            
                            weights_file = 'weights/weights_tmin_{:.5f}_tmax_{:.5f}_{}.json'.format(t_min, t_max, weights_key)
                            self.save_weights(self,weights_file)     
                            tf.print('Increase time interval ==> Epoch: {0:d}, total_loss: {1:.3e}, loss_BC: {2:.3e}, loss_IC: {3:.3e}, loss_f: {4:.3e}'.format(epoch, global_loss_Master, global_loss_BC_Master,global_loss_IC_Master, global_loss_f_Master))
                            tf.print("\n ")
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
                            flag_train_Master=0

        f_mem.close()
        return list_loss_workers
    ###############################################
