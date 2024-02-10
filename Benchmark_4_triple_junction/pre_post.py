#import tensorflow as tf
import datetime, os

import scipy.optimize
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
import tensorflow as tf

class PrePost:
    def __init__(self,X ,T, lb, ub,Nx,Ny,dx,dy,x,y, eta,phi_true):
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
        self.dx= dx
        self.dy=dy
        self.eta=eta
        self.phi_true=phi_true  
        self.num_phases=4
        #self.precision=tf.float64
        
    ###############################################
    def filter_X_phi_test(self,X_phi_test, save_npz=False):        
        # Replace subset_2 by random binary values
        for i in range(len(X_phi_test)):
            subset_1 = X_phi_test[i, :3]
            subset_2 = X_phi_test[i, 3:].astype(int)

            # Generate random binary values for subset_2
            subset_2 = np.random.randint(0, 2, size=len(subset_2))

            # Check if subset_2 matches the conditions
            while np.all(subset_2 == 0) or np.all(subset_2 == 1):
                subset_2 = np.random.randint(0, 2, size=len(subset_2))

            X_phi_test[i, 3:] = subset_2 
            if save_npz:
                np.savez('X_phi_test.npz',X_phi_test)
        return X_phi_test
    ###############################################
    def get_fields_x_y(self,all_phases_indexes,loc_index):
        Nx=self.Nx
        Ny=self.Ny
        phase_fields_x = np.zeros((Nx, Ny))
        phase_fields_y = np.zeros((Nx, Ny))
        grain= np.zeros((Nx, Ny))
        loc_index_0 = all_phases_indexes[0]
        loc_index_1 = all_phases_indexes[1]
        loc_index_2 = all_phases_indexes[2]
        loc_index_3 = all_phases_indexes[3]
        
        for i in range(Nx):
            for j in range(Ny):
                if i < Nx // 2:
                    phase_fields_x[i, j] = loc_index_0
                    phase_fields_y[i, j] = loc_index_0
                else:
                    phase_fields_x[i, j] = loc_index_1
                    phase_fields_y[i, j] = loc_index_1
        for i in range(Nx):
            for j in range(Ny):
                if j < Ny // 4 or j >= Ny * 3 // 4:
                    if i < Nx // 4 or i >= Nx * 3 // 4:
                        phase_fields_x[i, j] = loc_index_2
                        phase_fields_y[i, j] = loc_index_2
                    else:
                        phase_fields_x[i, j] = loc_index_3
                        phase_fields_y[i, j] = loc_index_3
        
        for i in range(Nx):
            for j in range(Ny):
                if phase_fields_x[i, j]==loc_index:
                    grain[i, j]=1
        return phase_fields_x, phase_fields_y, grain
    ###############################################
    def smooth_phase_field(self, phase_field, kernel_size=3, smoothing_factor=0.5):
        # Get the dimensions of the phase field matrix
        rows, cols = phase_field.shape
        
        # Create a new matrix for storing the smoothed values
        smoothed_field = np.zeros_like(phase_field)
        
        # Calculate the padding required for handling edges
        padding = kernel_size // 2
        
        # Iterate over each element in the phase field
        for i in range(rows):
            for j in range(cols):
                # Get the neighbors within the kernel size
                neighbors = phase_field[max(0, i-padding):min(rows, i+padding+1),
                                        max(0, j-padding):min(cols, j+padding+1)]
                
                # Check if the current element is a grain value
                if phase_field[i, j] == 1:
                    # If it is a grain value, assign it directly to the smoothed field
                    smoothed_field[i, j] = phase_field[i, j]
                else:
                    # If it is not a grain value, calculate the average of the neighbors
                    average = np.mean(neighbors)
                    
                    # Calculate the weighted average between the current element and the average
                    weighted_avg = (1 - smoothing_factor) * phase_field[i, j] + smoothing_factor * average
                    
                    # Assign the weighted average value to the corresponding position in the smoothed field
                    smoothed_field[i, j] = weighted_avg
        
        return smoothed_field
    ###############################################
    ###############################################
    ###############################################
    def get_Phases_Indexes(self,interface_width,int_gid_points,all_phases_indexes):
        Nx=self.Nx
        Ny=self.Ny
        phase_fields = np.zeros((Nx, Ny), dtype=int)
        phase_flags = np.full((Nx, Ny), 2, dtype=int)
        phase_fields_x = np.zeros((Nx, Ny))
        phase_fields_y = np.zeros((Nx, Ny))
        
        loc_index_0 = all_phases_indexes[0]
        loc_index_1 = all_phases_indexes[1]
        loc_index_2 = all_phases_indexes[2]
        loc_index_3 = all_phases_indexes[3]
              
        for i in range(Nx):
            for j in range(Ny):
                if i <= Nx // 2-1 :
                    phase_fields_x[i, j] = loc_index_0
                    phase_fields_y[i, j] = loc_index_0
                else:
                    phase_fields_x[i, j] = loc_index_1
                    phase_fields_y[i, j] = loc_index_1

        for i in range(Nx):
            for j in range(Ny):
                if j < Ny // 4 or j > Ny * 3 // 4:
                    if i <= Nx // 4 or i >= Nx * 3 // 4:
                        phase_fields_x[i, j] = loc_index_2
                        phase_fields_y[i, j] = loc_index_2
                    else:
                        phase_fields_x[i, j] = loc_index_3
                        phase_fields_y[i, j] = loc_index_3
         
        ################ Indexes #########################
        for i in range(Ny - 1):
            for j in range(Nx ):                   
                if phase_fields_x[i, j] != phase_fields_x[i + 1, j] and phase_fields_x[i, j]%1==0 and phase_fields_x[i+1, j]%1==0 :
                    loc_phase_1=phase_fields_x[i, j]
                    loc_phase_2=phase_fields_x[i+1, j] 
                    start_int_x=int(i)-1
                    #tf.print("loc_phase_1",loc_phase_1)
                    #tf.print("loc_phase_2",loc_phase_2)
                    #tf.print("start_int_x",start_int_x)
                    for int_grid_point in range(int_gid_points):
                        #tf.print("start_int_x+int_grid_point,j",start_int_x+int_grid_point,j)
                        
                        new_loc_phase=loc_phase_1 + (loc_phase_2-loc_phase_1)/int_gid_points* int_grid_point
                        #tf.print("new_loc_phase: ",new_loc_phase)
                        phase_fields_x[start_int_x+int_grid_point, j] =new_loc_phase
        
        #plt.imshow(phase_fields_x,origin='lower')
        #plt.xlabel("j")
        #plt.ylabel("i")
        #plt.show()    

        phase_fields_y_l=np.copy(phase_fields_x)        
        for i in range(Ny):
            for j in range(Nx-1):                 
                if phase_fields_y_l[i, j] != phase_fields_y_l[i , j+1] and phase_fields_y_l[i, j]%1==0  :
                    loc_phase_1=phase_fields_y_l[i, j]
                    loc_phase_2=phase_fields_y_l[i, j+1] 
                    #print(i,j)
                    
                    if phase_fields_y_l[i, j] < phase_fields_y_l[i , j+1]:
                        start_int_x=int(j)-1
                        for int_grid_point in range(int_gid_points):
                            #tf.print("start_int_x+int_grid_point,j",start_int_x+int_grid_point,j)
                            new_loc_phase=loc_phase_1 + (loc_phase_2-loc_phase_1)/int_gid_points* int_grid_point
                            #tf.print("new_loc_phase: ",new_loc_phase)
                            phase_fields_y_l[i, start_int_x+int_grid_point] =new_loc_phase  

        #plt.imshow(phase_fields_y_l)
        #plt.show()
        
        
        flag=1
        phase_fields_y_r=np.copy(phase_fields_x)   
        for i in range(Ny):
            for j in range(Nx-1):  
                if phase_fields_y_r[i, j] != phase_fields_y_r[i , j+1] and phase_fields_y_r[i, j+1]%1==0 :
                    
                    loc_phase_1=phase_fields_y_r[i, j]
                    loc_phase_2=phase_fields_y_r[i, j+1] 


                    if phase_fields_y_r[i, j] > phase_fields_y_r[i , j+1] : #and (j==idx_j):# or phase_fields_y_r[0, j]%1)==0 :
                        if flag:
                            idx_j=j
                            idx_i=i
                            #print(idx_i,idx_j)
            
                            flag=0
                        
                        if j==idx_j:                     
                            start_int_x=int(j)-1
                            for int_grid_point in range(int_gid_points):
                                #tf.print("start_int_x+int_grid_point,j",start_int_x+int_grid_point,j)
                                new_loc_phase=loc_phase_1 + (loc_phase_2-loc_phase_1)/int_gid_points* int_grid_point
                                #tf.print("new_loc_phase: ",new_loc_phase)
                                if start_int_x+int_grid_point<Nx:
                                    phase_fields_y_r[i, start_int_x+int_grid_point] =new_loc_phase  

        
        
        #plt.imshow(phase_fields_y_r,origin='lower')
        #plt.title("A")
        #plt.xlabel("j")
        #plt.ylabel("i")
        #plt.show()        
        
        
        
        #phase_fields_y=np.copy(phase_fields_y) 

                            
        #phase_fields_x=self.smooth_phase_field(phase_fields_x, kernel_size=1, smoothing_factor=0.75)
        #plt.imshow(phase_fields_y)
        #plt.show()
   
        """
        for i in range(Ny-1):
            for j in range(Nx-1):     
                if (phase_fields_y[i, j]%1==0 ) and (phase_fields_y[i, j] > phase_fields_y[i , j+1]) and (phase_fields_y[i, j] > phase_fields_y[i+1 , j]) and phase_fields_y[i, j] != phase_fields_y[i , j+1]  and phase_fields_y[i+1, j+1]==0  :
                    print("here",i,j)
                    loc_phase_1=phase_fields_y[i, j]
                    loc_phase_2=phase_fields_y[i+1, j+1]  

                    start_int=int(j)
                    print(loc_phase_1,loc_phase_2)
                    for int_grid_point_x in range(int_gid_points-1): 
                        for int_grid_point_y in range(int_gid_points-1): 
                            new_loc_phase_x=loc_phase_1 + (loc_phase_2-loc_phase_1)/(int_gid_points)* (int_grid_point_x)
                            new_loc_phase_y=loc_phase_1 + (loc_phase_2-loc_phase_1)/(int_gid_points)* (int_grid_point_y)
                            phase_fields_y[i+int_grid_point_x+1,start_int+int_grid_point_y+1] = np.sqrt(new_loc_phase_x**2+new_loc_phase_y**2)
        """                        
        #phase_fields_x=self.smooth_phase_field(phase_fields_x, kernel_size=1, smoothing_factor=0.75)

        #plt.imshow(phase_fields_x)
        #plt.show()
                
        #plt.imshow(phase_fields_y_l+phase_fields_y_r-phase_fields_x)
        #plt.show()
        
        #plt.imshow(phase_fields_y_r)
        #plt.show()
                

        #phase_fields_y=self.smooth_phase_field(phase_fields_y, kernel_size=1, smoothing_factor=0.75)

        #plt.imshow((0*phase_fields_x+phase_fields_y))
        #plt.show()

        phase_fields=phase_fields_y_l+phase_fields_y_r-phase_fields_x
        
        #plt.imshow(phase_fields,origin='lower')
        #plt.title("A")
        #plt.xlabel("j")
        #plt.ylabel("i")
        #plt.show()   
         
        return phase_fields #
   ###############################################
    def plot_phase_fields(self,phase_fields, title,all_phases_indexes=None,path=None,flag=False):

        plt.imshow(phase_fields, cmap='jet', interpolation=None, origin='lower')
        plt.xlabel('x')
        plt.ylabel('y')

        if flag:    
            for phase_index in range(len(all_phases_indexes)):
                indices = np.argwhere(phase_fields == phase_index)
                if len(indices) > 0:
                    i, j = indices[0]
                    plt.text(j, i, str(all_phases_indexes[phase_index]), color='red', ha='center', va='center')

        plt.colorbar()
        save_path = os.path.join(path, title)
       
        plt.title(title)
        plt.savefig(save_path)
        plt.close()
    ###############################################
    ###############################################
    def initialize_phases(self,all_phases_indexes,pathInput):
    ###############################################
    ###############################################
        Nx = self.Nx
        Ny = self.Ny
        
        phase_fields= np.zeros((Nx, Ny))

        all_phases_fractions=np.zeros((4,Nx, Ny))

        phase_flags = np.full((Nx, Ny), 2)

        interface_width = int(self.eta)  # Convert interface width to an integer
        int_gid_points=int(self.eta/self.dx)
        
        ### Phase Indexes
        phases_indexes= self.get_Phases_Indexes(interface_width,int_gid_points,all_phases_indexes)
        
        self.plot_phase_fields(phases_indexes,"Phase_Indexes",all_phases_indexes,pathInput,flag=True)

        sum_phases= np.zeros((Nx, Ny))     
        sum_interfaces= np.zeros((Nx, Ny))  
        sum_phases_and_interfaces= np.zeros((Nx, Ny))  

        all_flags_martrix=np.zeros((len(all_phases_indexes),Nx, Ny))  
        all_phases=np.zeros((len(all_phases_indexes),Nx, Ny)) 
        all_interfaces=np.zeros((len(all_phases_indexes),Nx, Ny))  
        
        for idx_phase in range(len(all_phases_indexes)): #
            interface= np.zeros((Nx, Ny))
            idx_phase_ = all_phases_indexes[idx_phase]
            flag = all_flags_martrix[idx_phase]
            
            new_indexes = [1 if index == idx_phase_ else 0 for index in all_phases_indexes]

            phase= self.get_Phases_Indexes(interface_width,int_gid_points,new_indexes)

            for i in range(Nx):
                for j in range(Ny):
                    if 0 < phase[i, j] < 1:
                        interface[i, j] = phase[i, j]
                    if 0 < phase[i, j] :
                        flag[i, j] =1

            all_flags_martrix[idx_phase]=flag
            all_phases[idx_phase]=phase
            all_interfaces[idx_phase]=interface

            self.plot_phase_fields(phase,  f'Phase_{idx_phase}',new_indexes,pathInput)
            self.plot_phase_fields(interface,  f'Interface_Phase_{idx_phase}',new_indexes,pathInput)
            self.plot_phase_fields(flag,  f'Interface_Flag_{idx_phase}',new_indexes,pathInput)
            
            sum_phases+=phase
            sum_interfaces+=interface
            sum_phases_and_interfaces+= (phase+interface)
            
        """
        plt.imshow(all_phases[0],origin='lower')
        plt.title("AB")
        plt.xlabel("j")
        plt.ylabel("i")
        plt.colorbar()
        plt.show()   
                
        plt.imshow(all_phases[1],origin='lower')
        plt.title("AB")
        plt.xlabel("j")
        plt.ylabel("i")
        plt.colorbar()
        plt.show()   

        plt.imshow(all_phases[2],origin='lower')
        plt.title("AB")
        plt.xlabel("j")
        plt.ylabel("i")
        plt.colorbar()
        plt.show() 
        
        plt.imshow(all_phases[3],origin='lower')
        plt.title("AB")
        plt.xlabel("j")
        plt.ylabel("i")
        plt.colorbar()
        plt.show() 
        """
        junctions = all_flags_martrix[0]+all_flags_martrix[1]+all_flags_martrix[2]+all_flags_martrix[3]

        self.plot_phase_fields(sum_phases,"Sum_Phases",new_indexes,pathInput)
        self.plot_phase_fields(sum_interfaces,"Sum_Interfaces",new_indexes,pathInput)
        self.plot_phase_fields(junctions,"Junctions",new_indexes,pathInput)
        self.plot_phase_fields(sum_phases_and_interfaces,  'Sum_Phases_&_interfaces',new_indexes,pathInput)
        
        #from sklearn.preprocessing import MinMaxScaler

        #scaler = MinMaxScaler(feature_range=(0, 1))
        #normalized_sum_phases = scaler.fit_transform(sum_phases)  # Normalize using MinMaxScaler

        X_ini_all = phases_indexes
        return phases_indexes,  all_flags_martrix, all_phases, all_interfaces
    ###################################
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
    def init_micro_cir(self):
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
        x = X_ini_all[0][:, 0]
        y = X_ini_all[0][:, 1]
        #tf.print("X_ini_all[0]",X_ini_all[0].shape)

        im1 = axs[0].imshow(X_ini_all[0], cmap=plt.get_cmap('jet'), interpolation='none')#,extent=[0,10,0,10])
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('y')
        axs[0].set_title('X_ini_all')
        #axs[0].invert_yaxis()  # Invert the y-axis to match imshow
        cbar = fig.colorbar(im1, ax=axs[0],shrink=0.35)

        # plot phi
        im2 = axs[1].imshow(phi_0[0], cmap=plt.get_cmap('jet'), interpolation='none')#,extent=[0,10,0,10])
        axs[1].set_title(r'$\phi$_ini')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('y')
        cbar = fig.colorbar(im2, ax=axs[1],shrink=0.35)
        plt.savefig(os.path.join(path,'micro_init'))
        plt.close()
     ###############################################
    def generate_random_column(self,N_f):
        while True:
            column = np.random.randint(0, 2, size=N_f)
            if np.any(column == 0) and np.any(column == 1):
                return column
    ##############################################
    def generate_random_columns(self,X_f_train, N_f, num_columns):
        for i in range(3, num_columns):
            X_f_train[:, i] = generate_random_column(N_f)
    ###############################################
    def set_training_data(self,x,y,N_ini,all_phases, all_interfaces,\
                          all_flags_martrix,N_f,tb,lb,ub,path):
                          
        f_values = [0, 0.33, 0.67, 1]
        '''Collocation Points'''
        # Latin Hypercube sampling for collocation points 
        X_f_train = self.lb + (self.ub - self.lb) * lhs(3, N_f)
        random_indices = np.random.choice(f_values, size=N_f)
        #X_f_train = np.column_stack((X_f_train, random_indices))
  
        X_ini_all=np.copy(X_f_train) 

        X_ini_all[:,2]=0
        All_phi_ini=all_phases
        All_flag_ini=all_flags_martrix
        All_interfaces_ini=all_interfaces

        X,Y = np.meshgrid(x,y)
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        
        
        X_ini_all=[]
        phi_ini_all=[]
        fig, axs = plt.subplots(self.num_phases, 2,figsize=(20, 30))
        fig.subplots_adjust(hspace=0.8, wspace=0.4)
        
        for idx in range(self.num_phases):
            X_mat = np.hstack((X_flat.reshape(-1, 1), Y_flat.reshape(-1, 1)))
            phi_mat = np.zeros((len(X_mat), self.num_phases))
            
            # Get the phi and flag arrays for the current phase
            phi = All_phi_ini[idx].flatten()
            flag = All_flag_ini[idx].flatten()
            #print("flag: ",flag.shape)

            # Stack additional columns horizontally
            zero_column = np.zeros((len(X_mat), 1))
            flag_column = np.array([f_values[idx]] * len(X_mat))[:, np.newaxis]
            X_mat = np.hstack((X_mat, zero_column))            
            phi_mat[:,idx]=phi
            X_ini_all.append(X_mat)
            phi_ini_all.append(phi_mat)

            axs[idx,0].scatter(X_mat[:, 0], X_mat[:, 1], cmap=plt.get_cmap('viridis'), c=phi_mat[:,idx])
            axs[idx,0].set_title(f"Flag of Phase {idx}")

            axs[idx,1].scatter(X_mat[:, 0], X_mat[:, 1], cmap=plt.get_cmap('viridis'), c=phi_mat[:,idx])
            axs[idx,1].set_title(f"Phi of Phase {idx}")
            
        save_path = os.path.join(path, "X_ini_all_phi_ini_all") 
        plt.savefig(save_path)
        plt.close()
        X_ini_all = np.vstack(X_ini_all)  
        phi_ini_all = np.vstack(phi_ini_all)  
        phi_ini_all=np.sum(phi_ini_all, axis=1).reshape(-1, 1)



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
        # filter 
        positive_indices = []
        for i in range(len(phi_ini_all)):
            row = phi_ini_all[i]
            row_sum = np.sum(row)
            if row_sum >0:
                positive_indices.append(i)
                
        #X_ini_all =X_ini_all[positive_indices]
        #phi_ini_all=phi_ini_all[positive_indices]

        
        tf.print('X_f_train: {0}, X_ini_all: {1}, '
                'X_lb_train: {2}, X_ub_train: {3}, '
                'X_ltb_train: {4}, X_rtb_train: {5}, '
                'phi_ini_all: {6}'.format(X_f_train.shape,
                                        X_ini_all.shape,
                                        X_lb_train.shape,
                                        X_ub_train.shape,
                                        X_ltb_train.shape,
                                        X_rtb_train.shape,
                                        phi_ini_all.shape))
        
        return X_f_train, X_ini_all,X_lb_train,X_ub_train,\
            X_rtb_train,X_ltb_train,phi_ini_all
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
        axs[1].set_title('Sum_Interfaces_ini')
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
        weights_loaded = tf.cast(weights_loaded, dtype=self.precision)
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
    def plot_global_evolution_discret(self,num_boxes,X_phi_test_sub, phi_evolution,pathOutput,title,filename,t_max,Nt  ):
        
        box_size = len(phi_evolution)// num_boxes
        num_rows = (num_boxes + 1) // 2
        fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(8, 4*num_rows), constrained_layout=True)
        indices = np.around(np.linspace(0, len(phi_evolution) - 1, num_boxes)).astype(int)
        for i, ax in enumerate(axes.flat):
            phi=phi_evolution[indices[i]]
            #phi = np.clip(phi, 0, 1)
            im=ax.scatter(X_phi_test_sub[:, 0], X_phi_test_sub[:, 1], cmap=plt.get_cmap('viridis'), c=phi,vmin=0,vmax=1)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            if i==0:
                cbar = fig.colorbar(im, ax=ax, shrink=0.5)
                cbar.ax.set_ylabel(r'$\phi$')
            time=indices[i]/Nt
            percentage = time * 100
            ax.set_title(f'φ at Time: {percentage:.2f}%')
            
        
        plt.savefig(os.path.join(pathOutput ,filename))
        plt.close()
    ###############################################
    def plot_global_evolution_continous(self,num_boxes, phi_evolution,pathOutput,title,filename,t_max ):
        
        import pickle
        """
        # PF solution
        with open('plot_data_PF.pickle', 'rb') as f:
            loaded_plot_data = pickle.load(f)
        raduis_PF = loaded_plot_data
        
        # FD solution 
        with open('plot_data_FD.pickle', 'rb') as f:
            loaded_plot_data = pickle.load(f)
        raduis_FD = loaded_plot_data
        """

        box_size = len(phi_evolution)// num_boxes
        num_rows = (num_boxes + 1) // 2

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
                phi = np.clip(phi, 0, 1)
                im=ax.imshow(phi, cmap='jet', interpolation='none', vmin=0, vmax=1)
                if i==0:
                    cbar = fig.colorbar(im, ax=ax, shrink=0.5)
                    cbar.ax.set_ylabel(r'$\phi$')
                ax.set_xlabel('x',fontsize=18)
                ax.set_ylabel('y',fontsize=18)
                ax.tick_params(axis='x', labelsize=18)
                ax.tick_params(axis='y', labelsize=18)
                percentage = indices[i] / len(phi_evolution) * 100
                ax.set_title(f'φ at Time: {percentage:.2f}%',fontsize=18)
                t=i * box_size
                
            else:
                #print("len(phi_evolution): ",len(phi_evolution)) 
                for t in range(len(phi_evolution)):
                    
                    phi=phi_evolution[t]
                    if t==0:
                        area_vs_t=len(phi[phi>1e-3])
                    else:
                        area_vs_t=len(phi[phi>thresh])
                    out_area_vs_t.append([t,area_vs_t])
                        
                out_radius_vs_t=np.sqrt(np.asarray(out_area_vs_t)[:,1] /(self.Nx*self.Ny)/np.pi)
                # Add the area vs. time plot to the last subplot
                ax.plot(out_time,out_radius_vs_t[::], "r--",label=r"$PINN$")
                ax.set_xlabel('Time (dimensionless)',fontsize=18)
                ax.set_xlim([0,t_max])
                ax.set_ylabel('Radius',fontsize=18)
                ax.set_title('Radius vs. Time',fontsize=18)
                ax.tick_params(axis='x', labelsize=18)
                ax.tick_params(axis='y', labelsize=18)
                ax.legend(fontsize=12)
                break
            
        fig.suptitle(title)
 
        plt.savefig(os.path.join(pathOutput ,filename))
        plt.close()
    #########################################################
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
            ax.set_title(f'$t = {np.float32(t[time]):.2f}$', fontsize=10)
            if i == 1:
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
                

         #plt.savefig('results.png', dpi=500)

    
