import tensorflow as tf
import datetime, os
#hide tf logs 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'} 
#0 (default) shows all, 1 to filter out INFO logs, 2 to additionally filter out WARNING logs, and 3 to additionally filter out ERROR logs
import scipy.optimize
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import time
from pyDOE import lhs         #Latin Hypercube Sampling
import seaborn as sns 
import codecs, json
import math
# generates same random numbers each time
import random
import datetime
import shutil



class PrePost:
    def __init__(self,X ,T, lb, ub, x, eta):
        """
        Initialize instance variables here.
        """
        self.X=X
        self.T=T
        self.lb = lb
        self.ub = ub
        self.x= x
        self.eta=eta

    ###############################################
    def set_training_data(self,N_ini,N_f,tb,eta,x):

        '''Boundary Conditions'''
        #Initial Condition -1 =< x =<1 and t = 0  
        all_x_IC= np.hstack((self.X[0,:][:,None], self.T[0,:][:,None])) 
        all_phi_IC = np.zeros_like(x)
        all_phi_IC[x < -eta/2] = 1.0
        all_phi_IC[(x >= -eta/2) & (x <= eta/2)] = 0.5*(1-np.sin(np.pi*x[(x >= -eta/2) & (x <= eta/2)]/eta))
        all_phi_IC[x > eta/2] = 0.0
 

        idx_ini = np.asarray(sorted(np.random.choice(all_x_IC.shape[0], N_ini,replace=False))) 
        X_ini_train = all_x_IC[idx_ini,:] #choose indices from  set 'idx' (x,t)
        phi_ini_train = all_phi_IC[idx_ini,:] #choose indices from  set 'idx' (x,t)

        '''Collocation Points'''
        # Latin Hypercube sampling for collocation points 
        # N_f sets of tuples(x,t)
        X_f_train = self.lb + (self.ub-self.lb)*lhs(2,N_f)

        #B oundary Condition x = -1 and 0 =< t =<1 
        X_lb_train = np.concatenate((0*tb + self.lb[0], tb), 1)

        #Boundary Condition x = 1 and 0 =< t =<1
        X_ub_train = np.concatenate((0*tb + self.ub[0], tb), 1)
        #X_f_train = np.vstack((X_f_train, X_u_train)) # append training points to collocation points 

        X_ini_train[:, 1] = 0

        print('X_f_train: {0}, X_ini_train: {1}, X_lb_train: {2}, X_ub_train: {3}, u_ini_train: {4}'.format(X_f_train.shape,\
        X_ini_train.shape,X_lb_train.shape,X_ub_train.shape,phi_ini_train.shape))
            
        return X_f_train, X_ini_train,X_lb_train,X_ub_train,phi_ini_train
    
    ###############################################
    def plot_Collocation_IC_BC(self,X_ini_train,X_f_train,X_lb_train,X_ub_train):
        # Training datafig,ax = plt.subplots()

        # plt.plot(X_obs_train[:,1], X_obs_train[:,0], '*', color = 'red', markersize = 5, label = 'Boundary collocation = 100')
        plt.plot(X_ini_train[:,1], X_ini_train[:,0], '*', color = 'b', markersize = 5, label = 'IC :initial condition')
        plt.plot(X_f_train[:,1], X_f_train[:,0], 'o', markersize = 0.5, label = 'PDE Collocation')
        plt.plot(X_lb_train[:,1], X_lb_train[:,0], '*', color = 'g', markersize = 5, label = 'BC: Bottom edge')
        plt.plot(X_ub_train[:,1], X_ub_train[:,0], '*', color = 'm', markersize = 5, label = 'BC: Upper edge')

        plt.xlabel('t')
        plt.ylabel('x')
        plt.title('Collocation Points')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

        #fig.savefig('collocation_points.png', dpi = 500, bbox_inches='tight')
    def plot_domain(self,X_ini_train,X_f_train,X_ub_train,X_lb_train,t_min, t_max,batch_idx):
        pathOutput = os.getcwd() + '/save_figs'
        plt.figure()
        plt.plot(X_ini_train[:,1], X_ini_train[:,0], '*', color = 'b', markersize = 5, label = 'IC :initial condition')
        plt.plot(X_f_train[:,1], X_f_train[:,0], 'o', markersize = 0.5, label = 'PDE Collocation')
        plt.plot(X_lb_train[:,1], X_lb_train[:,0], '*', color = 'g', markersize = 5, label = 'BC: Bottom edge')
        plt.plot(X_ub_train[:,1], X_ub_train[:,0], '*', color = 'm', markersize = 5, label = 'BC: Upper edge')

        plt.xlabel('t')
        plt.ylabel('x')
        plt.title('Collocation Points')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        title="Training domain for Batch "+ str(batch_idx)+" for: " +'t_min: {0:.3f}, t_max: {1:.3f}'.format(t_min, t_max)
        plt.savefig(os.path.join(pathOutput ,title+ ".png"))
        plt.close()

    ###############################################
    def plot_loss(self,list_loss):
        list_loss=np.asarray(list_loss)
        fig,ax=plt.subplots()
        ax.plot(list_loss[:, 0], label='total_loss')
        ax.plot(list_loss[:, 1], label='loss_BC')
        ax.plot( list_loss[:, 2], label='loss_IC')
        ax.plot(list_loss[:, 3], label='loss_f')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        plt.show()

    ###############################################
    def solution_plot(self,u_pred,phi_sol):


        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axis('off')
        #######  gs0 
        gs0 = gridspec.GridSpec(1, 2)
        gs0.update(top=1, bottom=0.8, left=0.15, right=0.85, wspace=0)
        ax = plt.subplot(gs0[:, :])
        h = ax.imshow(u_pred, interpolation='nearest', cmap='rainbow', 
                    extent=[self.T.min(), self.T.max(), self.X.min(), self.X.max()], 
                    origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
        line = np.linspace(self.X.min(), self.X.max(), 2)[:,None]
    

        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.legend(frameon=False, loc = 'best')
        ax.set_title(r'$\phi(x,t)$', fontsize = 10)

        ####### gs1
        ''' 
        Slices of the solution at regular intervals (n_times)
        '''
        gs1 = gridspec.GridSpec(2, 4)
        gs1.update(top=0.8, bottom=0.5, left=0.1, right=0.9, wspace=0.5)

        n_times = 4
        time_indices = np.linspace(0, len(u_pred) - 1, n_times).astype(int)
        x=self.X[0]
        t=np.linspace(self.T.min(), self.T.max(), len(u_pred))
        for i, time in enumerate(time_indices):
            ax = plt.subplot(gs1[1, i])
            ax.plot(x,phi_sol[time,:], 'b--', linewidth = 2, label = 'Exact')
            ax.plot(x,u_pred.T[time,:], 'r--', linewidth = 2, label = 'Prediction')
            ax.set_xlabel('$x$')
            ax.set_ylabel('$\phi(x,t)$')
            ax.axis('square')
            ax.set_xlim([-1.1,1.1])
            ax.set_ylim([0,1.1])
            ax.set_title(f'$t = {np.float64(t[time]):.2f}$', fontsize=10)
            if i == 1:
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
                

         #plt.savefig('results.png', dpi=500)

    
