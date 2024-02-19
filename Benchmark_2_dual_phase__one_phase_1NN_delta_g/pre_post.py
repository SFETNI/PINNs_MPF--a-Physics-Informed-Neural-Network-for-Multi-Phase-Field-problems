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
import glob 
import re 

class PrePost:
    def __init__(self,X ,T, lb, ub,Nx,Ny, x,y, eta,phi_true,R0):
        """
        Initialize instance variables here.
        """
        self.X=X
        self.T=T
        self.lb = lb
        self.ub = ub
        self.Nx = Nx
        self.Ny = Ny
        self.x= x
        self.y=y
        self.eta=eta
        self.phi_true=phi_true  
        self.R0=R0 
    ###############################################
    def get_phi_values(self,r,i_cir):
        phi_inf = 0.0
        phi_mid = 0.5 - 0.5*np.sin(np.pi*(r-self.R0[i_cir])/self.eta)
        phi_sup = 1.0
        
        inf = r < self.R0[i_cir] - self.eta/2
        mid = np.logical_and(r >= self.R0[i_cir] - self.eta/2, r <= self.R0[i_cir] + self.eta/2)
        sup = r > self.R0[i_cir] + self.eta/2
    
        phi_values = np.where(inf, phi_sup, np.where(mid, phi_mid, phi_inf))
        return phi_values   
    ###############################################
    def get_X_ini_values(self,r,i_cir):
        phi_inf = 0.0
        phi_mid = 0.5 - 0.5*np.sin(np.pi*(r-self.R0[i_cir])/self.eta)
        phi_sup = 1.0
        
        inf = r < self.R0[i_cir] - self.eta/2
        mid = np.logical_and(r >= self.R0[i_cir] - self.eta/2, r <= self.R0[i_cir] + self.eta/2)
        sup = r > self.R0[i_cir] + self.eta/2
    
        X_ini_all_values = np.where(inf, phi_sup, np.where(mid, phi_mid, phi_inf))
        return X_ini_all_values   
    ###############################################
    def init_micro_cir(self,ox,oy,oz,N_ini,Nx,Ny,x,y,lb,ub):
        all_phi     = np.zeros((Nx,Ny))
        X_ini_all=[]
        xcor_linspace  = np.linspace(x.min(),x.max(),Nx)
        ycor_linspace  = np.linspace(y.min(),y.max(),Ny)
        zcor_linspace  = np.zeros((Nx**2))   
    
        #xcor,ycor = np.meshgrid(xcor_linspace,ycor_linspace)
        for i_cir in range(len(self.R0)): 
            x_center=ox[i_cir]
            y_center=oy[i_cir]
            for i_coory in range(Ny):
                for i_coorx in range(Nx):
                        r=np.sqrt((x_center-xcor_linspace[i_coorx])**2
                            +(y_center-ycor_linspace[i_coory])**2)
                            #+(oz[i_cir]-zcor[i_coor])**2)
                        # phi(x,y,z) value ==> cases
                        phi=self.get_phi_values(r,i_cir)
                        if phi>0:
                            all_phi[i_coorx][i_coory]=phi
                            # X_ini_all are all points inside the cercle R0
                            X_ini_all.append([i_coorx,i_coory])
        X_ini_all=np.asarray(X_ini_all)
        X_ini_square = np.zeros((Nx, Ny))
        #plt.scatter(X_ini_all[:, 0], X_ini_all[:, 1])
        #plt.show()
        X_ini_square[X_ini_all[:, 0], X_ini_all[:, 1]] = all_phi[X_ini_all[:, 0], X_ini_all[:, 1]]

        # set lb and ub values for each feature
        #plt.imshow(X_ini_square)
        #plt.show()
        #plt.imshow(all_phi)
        #plt.show()
        return all_phi, X_ini_square #np.asarray(X_ini_all)
    ###############################################
    def plot_init(self,X_ini_all,phi_0,Nx,Ny,path=None):
        fig, axs = plt.subplots(1, 2, figsize=(10,10))
        # plot X_ini_all
        x = X_ini_all[:, 0]
        y = X_ini_all[:, 1]

        im1 = axs[0].imshow(X_ini_all, cmap=plt.get_cmap('jet'), interpolation='none')#,extent=[0,10,0,10])
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('y')
        axs[0].set_title('X_ini_all')
        #axs[0].invert_yaxis()  # Invert the y-axis to match imshow
        cbar = fig.colorbar(im1, ax=axs[0],shrink=0.35)

        # plot phi
        im2 = axs[1].imshow(phi_0, cmap=plt.get_cmap('jet'), interpolation='none')#,extent=[0,10,0,10])
        axs[1].set_title(r'$\phi$_ini')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('y')
        cbar = fig.colorbar(im2, ax=axs[1],shrink=0.35)
        plt.savefig(os.path.join(path,'micro_init'))
        plt.close()
    ###############################################
    def set_training_data(self,x,y,X_ini_all,N_ini,phi_0,N_f,tb,lb,ub):
        '''Collocation Points'''
        # Latin Hypercube sampling for collocation points 
        # N_f sets of tuples(x,t)
        X_f_train = self.lb + (self.ub-self.lb)*lhs(3,N_f)

        X_ini_all = (X_ini_all- np.min(X_ini_all)) / (np.max(X_ini_all) - np.min(X_ini_all))

                # Select positive ones, zeros, and interface indices from X_ini_all
        positive_ones_indices = np.argwhere(X_ini_all == 1)
        zeros_indices = np.argwhere(X_ini_all == 0)
        interface_indices = np.argwhere((X_ini_all > 0) & (X_ini_all < 1))

        #tf.print("shapes: ",positive_ones_indices.shape,zeros_indices.shape,interface_indices.shape,phi_0.shape)
        # Select random indices from each group (with percentages)
        n_zeros_indices = int(0.2* N_ini)
        n_positive_ones_indices = int(0.2 * N_ini)
        n_interface_indices = N_ini - n_zeros_indices - n_positive_ones_indices

        idx_zeros = np.random.choice(zeros_indices.shape[0], n_zeros_indices, replace=True)
        idx_positive_ones = np.random.choice(positive_ones_indices.shape[0], n_positive_ones_indices, replace=True)
        idx_interface = np.random.choice(interface_indices.shape[0], n_interface_indices, replace=True)

        # Get the corresponding points from X_ini_all
        X_ini_train_zeros = zeros_indices[idx_zeros]
        X_ini_train_positive_ones = positive_ones_indices[idx_positive_ones]
        X_ini_train_interface = interface_indices[idx_interface]
        #tf.print("shapes: ",X_ini_train_zeros.shape,X_ini_train_positive_ones.shape,X_ini_train_interface.shape )

        # Get the corresponding submatrices from phi_0
        phi_ini_train_zeros = phi_0[X_ini_train_zeros[:, 0], X_ini_train_zeros[:, 1]]
        phi_ini_train_positive_ones = phi_0[X_ini_train_positive_ones[:, 0], X_ini_train_positive_ones[:, 1]]
        phi_ini_train_interface = phi_0[X_ini_train_interface[:, 0], X_ini_train_interface[:, 1]]

        # Concatenate the data for training
        X_ini_train = np.concatenate((X_ini_train_zeros, X_ini_train_positive_ones, X_ini_train_interface), axis=0)
        phi_ini_train = np.concatenate((phi_ini_train_zeros, phi_ini_train_positive_ones, phi_ini_train_interface), axis=0)
        #phi_ini_train = phi_ini_train.reshape(-1, 1)

        # Get all IC points and corresponding phi values for subdomain decomposition/minibatching
        X_ini_train_all = np.concatenate((zeros_indices, positive_ones_indices, interface_indices), axis=0)
        phi_ini_train_all = phi_0[X_ini_train_all[:, 0], X_ini_train_all[:, 1]]
        #phi_ini_train_all = phi_ini_train_all.reshape(-1, 1)

        #Boundary Condition lower bound      
        X_lb_train = np.concatenate((np.linspace(x.min(), x.max(), tb.shape[0])[:, None],\
                                     np.repeat(self.lb[0], tb.shape[0])[:, None],\
                                          tb),\
                                              1)
   
        #Boundary Condition upper bound
        X_ub_train = np.concatenate((np.linspace(x.min(), x.max(), tb.shape[0])[:, None],\
                                     np.repeat(self.ub[1], tb.shape[0])[:, None],\
                                          tb), \
                                            1)
  
        #Boundary Condition right bound
        X_rtb_train = np.concatenate((np.repeat(self.ub[0], tb.shape[0])[:, None],\
                                      np.linspace(y.min(), y.max(), tb.shape[0])[:, None],\
                                          tb), \
                                            1)
   
        #Boundary Condition left bound
        X_ltb_train = np.concatenate((np.repeat(self.lb[1], tb.shape[0])[:, None],\
                                      np.linspace(y.min(), y.max(), tb.shape[0])[:, None],\
                                          tb), \
                                            1)
                
        # get the minimum and maximum values for each column
        min_values = 0
        max_values = self.Nx

        # scale the values between 0 and 1 with the reference range
        X_ini_train_scaled = lb[:2] + (X_ini_train - min_values) * (ub[:2] - lb[:2]) / (max_values - min_values)
        X_ini_train_all_scaled = lb[:2] + (X_ini_train_all - min_values) * (ub[:2] - lb[:2]) / (max_values - min_values)
        X_ini_train_scaled = np.hstack((X_ini_train_scaled, np.zeros((len(X_ini_train_scaled), 1))))
        X_ini_train_all_scaled = np.hstack((X_ini_train_all_scaled, np.zeros((len(X_ini_train_all_scaled), 1))))
        phi_ini_train=np.expand_dims(phi_ini_train, axis=1)
        phi_ini_train_all=np.expand_dims(phi_ini_train_all, axis=1)
        tf.print('X_f_train: {0}, X_ini_train: {1}, X_lb_train: {2}, X_ub_train: {3}, phi_ini_train: {4}'.format(X_f_train.shape,\
        X_ini_train_scaled.shape,X_lb_train.shape,X_ub_train.shape,phi_ini_train.shape))
        return X_f_train, X_ini_train_scaled,X_lb_train,X_ub_train,X_rtb_train,X_ltb_train,phi_ini_train, X_ini_train_all_scaled, phi_ini_train_all
    ###############################################
    def EraseFile(self,path=None):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    ###############################################
    def plot_Collocation_IC_BC(self,Nx,Ny,x,y,X_ini_train,X_f_train,X_lb_train,\
                               X_ub_train,X_rtb_train,X_ltb_train,phi_0,\
                                phi_ini_train,path=None,title ='Collocation_IC_BC_points',flag_train=False):
        fig, axs = plt.subplots(1, 2, figsize=(20,20))

        if (flag_train):
            tf.print("min max: ", x.min(),x.max(),y.min(),y.max())
            tf.print("shapes: ", X_f_train.shape, X_ini_train.shape,X_lb_train.shape,\
                               X_ub_train.shape,X_rtb_train.shape,X_ltb_train.shape )

        # plot X_ini_train and X_f_train
        scatter1 = axs[0].scatter(X_f_train[:, 0], X_f_train[:, 1], marker='o', label='PDE Collocation')
        scatter1.set_sizes([0.5])
        scatter2 = axs[0].scatter(X_ini_train[:, 0], X_ini_train[:, 1],cmap=plt.get_cmap('viridis') ,c=phi_ini_train,s=150)
        cbar = plt.colorbar(scatter2, ax=axs[0], shrink=0.35)
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('y')
        axs[0].set_title(f'X_ini_train ({X_ini_train.shape[0]} points) and X_f_train ({X_f_train.shape[0]} points)')
        axs[0].set_xlim([x.min(), x.max()])
        axs[0].set_ylim([y.min(), y.max()])
        axs[0].set_aspect("equal")
        #axs[0].invert_yaxis()
        axs[0].scatter(X_ub_train[:,0], X_ub_train[:,1], s=70, marker='*', color='m', label='BC: upper edge ('+str(X_ub_train.shape[0])+')')
        axs[0].scatter(X_lb_train[:,0], X_lb_train[:,1], s=70, marker='*', color='g', label='BC: lower edge ('+str(X_lb_train.shape[0])+')')
        axs[0].scatter(X_rtb_train[:,0], X_rtb_train[:,1], s=70, marker='*', color='orange', label='BC: right edge ('+str(X_rtb_train.shape[0])+')')
        axs[0].scatter(X_ltb_train[:,0], X_ltb_train[:,1], s=70, marker='*', color='r', label='BC: left edge ('+str(X_ltb_train.shape[0])+')')
        axs[0].legend()

        # plot phi
        im2 = axs[1].imshow(phi_0, cmap=plt.get_cmap('jet'), interpolation='none')#,extent=[0,10,0,10])
        axs[1].set_title(r'$\phi$_ini')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('y')
        cbar = fig.colorbar(im2, ax=axs[1],shrink=0.35)
        #axs[1].set_xlim([x.min(), x.max()])
        #axs[1].set_ylim([y.min(), y.max()])

        plt.savefig(os.path.join(path,title))
        plt.close()
    ###############################################
    def plot_domain(self,X_ini_train,X_ini_train_s,X_f_train,X_f_train_s,X_ub_train,X_ub_train_s,X_lb_train,\
                    X_lb_train_s,X_ltb_train,X_ltb_train_s,X_rtb_train,X_rtb_train_s,\
                    t_min, t_max,epoch,batch_idx,phi_0,phi_ini_train,phi_ini_train_s,path=None):
        title_domain = f"Training subdomain at Epoch {epoch} for Time interval: t_min: {t_min:.3f}, t_max: {t_max:.3f}.jpg"
        self.plot_Collocation_IC_BC(self.Nx,self.Ny,self.x,self.y,X_ini_train,X_f_train,X_lb_train,\
                                    X_ub_train,X_rtb_train,X_ltb_train,phi_0,phi_ini_train,path,title_domain,flag_train=False)
        title_batch = f"Training batch {batch_idx} at Epoch {epoch} for Time interval: t_min: {t_min:.3f}, t_max: {t_max:.3f}.jpg"
        self.plot_Collocation_IC_BC(self.Nx,self.Ny,self.x,self.y,X_ini_train_s,X_f_train_s,X_lb_train_s,\
                                    X_ub_train_s,X_rtb_train_s,X_ltb_train_s,phi_0,phi_ini_train_s,path,title_batch,flag_train=False)
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
    ##############################################
    def plot_exact(self,path=None):
        N = 4  # number of plots to create
        interval = len(self.phi_true) // N  # interval between plots
        times = np.arange(0, len(self.phi_true), interval)  # array of times for each plot

        plt.figure()
        for i, t in enumerate(times):
            label = 't_{}'.format(t) 
            plt.plot(self.x/self.eta, self.phi_true[t,:], '--', linewidth=2, label=label)
        label_last = 't_{}'.format(len(self.phi_true) - 1) + "_pred"
        plt.plot(self.x/self.eta, self.phi_true[-1,:], '-.', linewidth=2, label=label_last)     
        plt.legend()
        plt.xlabel(r'x/$\eta$')
        plt.ylabel(r'$\phi$')
        plt.title("Exact solution")

        plt.savefig(os.path.join(path,"Exact solution"))
        plt.close()
    ###############################################        
    def read_weights_files(self,path):
        weights_files = glob.glob(os.path.join(path, '*.json'))
        weights_files = sorted(weights_files)
        return weights_files
    ###############################################
    def load_weights(self,weights_file):
        with open(weights_file, 'r') as f:
            weights_loaded = json.load(f)['weights']
        weights_loaded = tf.cast(weights_loaded, dtype=tf.float64)
        return weights_loaded
    ###############################################
    def extract_t_min_t_max(self,filename):
        match = re.search(r'tmin_(\d+\.\d+)_tmax_(\d+\.\d+)', filename)
        if match:
            t_min = float(match.group(1))
            t_max = float(match.group(2))
            return t_min, t_max
        return None, None
    ###############################################
    def plot_global_evolution(self,num_boxes, phi_evolution,pathOutput,title,filename,t_max ):
        
        import pickle

        # Laden des Dictionaries aus der Datei
        with open('plot_data_FD.pickle', 'rb') as f:
            loaded_plot_data = pickle.load(f)
        raduis_FD = loaded_plot_data
        
        
        box_size = len(phi_evolution)// num_boxes
        # Compute the number of rows needed to display all subplots
        num_rows = (num_boxes + 1) // 2

        # Create the figure and subplots
        fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(8, 4*num_rows), constrained_layout=True)

        # Loop over the boxes and plot the corresponding phi values
        out_area_vs_t=[]
        out_radius_vs_t=[]
        thresh=1e-1
        indices = np.around(np.linspace(0, len(phi_evolution) - 1, num_boxes)).astype(int)

        for i, ax in enumerate(axes.flat):

            if i < num_boxes-1 :
                start_idx = i * box_size
                end_idx = (i + 1) * box_size
                #tf.print("start_idx: ",start_idx)
                phi=phi_evolution[indices[i]]
                import scipy.ndimage as ndimage
                
                phi = np.clip(phi, 0, 1)
                phi = ndimage.median_filter(phi, size=3)  # Apply median filter with kernel size 3
                #phi = phi[12:-12,12:-12]
                im=ax.imshow(phi, cmap='jet', interpolation='none', vmin=0, vmax=1)
                if i==0:
                    cbar = fig.colorbar(im, ax=ax, shrink=0.5)
                    cbar.ax.set_ylabel(r'$\phi$')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                percentage = indices[i] / len(phi_evolution) * 100
                ax.set_title(f'φ at Time: {percentage:.2f}%')
                t=i * box_size
                
            else:
                for t in range(len(phi_evolution)):
                    
                    phi=phi_evolution[t]
                    #phi = np.clip(phi, 0, 1)
                    phi = ndimage.median_filter(phi, size=3)
                    if t<=1:
                        area_vs_t=len(phi[phi>3.8e-2])
                    else:
                        area_vs_t=len(phi[phi>thresh])
                    
                    out_area_vs_t.append([t,area_vs_t])
                        
                out_radius_vs_t=np.sqrt(np.asarray(out_area_vs_t)[:,1] /(self.Nx*self.Ny)/np.pi)
                out_time=np.linspace(0,t_max,len(out_radius_vs_t))
                # Add the area vs. time plot to the last subplot
                ax.plot(out_time,out_radius_vs_t[::], "r--",label=r"$PINN disret$")
                ax.plot(np.linspace(0,1,len(raduis_FD)),raduis_FD, "b--",label=r"$Theory$" )
                ax.set_xlabel('Time')
                #ax.set_xlim([0,t_max])
                ax.set_ylabel('Radius')
                ax.set_title('Radius vs. Time')
                ax.legend()
                
                break
            
        fig.suptitle(title)
        plt.savefig(os.path.join(pathOutput ,filename))
        plt.close()
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
            ax.plot(x,u_pred.T[time,:],'r--', linewidth = 2, label = 'Prediction')
            ax.set_xlabel('$x$')
            ax.set_ylabel('$\phi(x,t)$')
            ax.axis('square')
            #ax.set_xlim([-1.1,1.1])
            #ax.set_ylim([0,1.1])
            ax.set_title(f'$t = {np.float64(t[time]):.2f}$', fontsize=10)
            if i == 1:
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
                
         plt.savefig('results.png', dpi=500)

    
