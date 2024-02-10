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
import psutil
from pyDOE import lhs         #Latin Hypercube Sampling
import seaborn as sns 
import codecs, json
import math
# generates same random numbers each time
np.random.seed(1234)
tf.random.set_seed(1234)
import random
import datetime
import shutil
import random
import glob 

from importlib import reload
import pre_post
reload(pre_post) 
from pre_post import *


class Sequentialmodel(tf.Module):
    ###############################################
    def __init__(self, layers, X_f_train, X_ini_train, X_lb_train, X_ub_train,\
                 X_ltb_train,X_rtb_train,phi_0,phi_ini_train,X_ini_train_all,phi_ini_train_all, N_ini, X_u_test, X,T,x,y,lb, ub, mu,\
                      sigma, delta_g, R0,\
                        X_center,Y_center,eta,Nx,Ny,Nt,phi_sol,name=None):
        super().__init__(name=name)

        self.X_f = X_f_train             
        self.X_ini = X_ini_train
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

        self.phi_0=phi_0
        self.phi_ini = phi_ini_train
        self.X_u_test = X_u_test
        self.X_ini_all_sub_domain=X_ini_train_all
        self.phi_ini_all_sub_domain=phi_ini_train_all


        self.X_ini_train_all=X_ini_train_all
        self.phi_ini_train_all=phi_ini_train_all
        self.N_ini =N_ini
        self.indices_ini = np.random.choice(len(self.X_ini_all_sub_domain), size=int(self.N_ini/2), replace=True)
        self.lb = lb
        self.ub = ub
        self.mu = mu
        self.sigma = sigma
        self.delta_g = delta_g
        self.R0=R0
        self.X_center=X_center
        self.Y_center=Y_center
        self.eta = eta
        self.layers = layers
        self.Nx=Nx
        self.Ny=Ny
        self.Nt=Nt
        self.x=x
        self.y=y
        self.lr=0.0001
        self.thresh=0
        
       

        self.abs_x_min=0  # to update and use during the minibatching (these points are the corners of the global space domain)
        self.abs_x_max=0
        self.abs_y_min =0
        self.abs_y_max=0      
        
        
        self.f=1
        self.ic=1
        self.bc=1   

        self.W = []  #Weights and biases
        self.parameters = 0 #total number of parameters
        
        for i in range(len(self.layers)-1):
            
            input_dim = layers[i]
            output_dim = layers[i+1]
            
            #Xavier standard deviation 
            std_dv = np.sqrt((2.0/(input_dim + output_dim)))

            #weights = normal distribution * Xavier standard deviation + 0
            w = tf.random.normal([input_dim, output_dim], dtype = 'float64') * std_dv
                       
            w = tf.Variable(w, trainable=True, name = 'w' + str(i+1))

            b = tf.Variable(tf.cast(tf.zeros([output_dim]), dtype = 'float64'), trainable = True, name = 'b' + str(i+1))
                    
            self.W.append(w)
            self.W.append(b)
            
            self.parameters +=  input_dim * output_dim + output_dim

        # Define the Adam optimizer
        self.optimizer_Adam = tf.keras.optimizers.Adam(learning_rate=self.lr, clipnorm=1) 
 
        self.PRE_POST=PrePost(X=X, T=T, lb=lb, ub=ub,Nx=self.Nx,Ny=self.Ny, x=x, y=y,eta=eta, phi_true=phi_sol,R0=R0)
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
    ###############################################
    def h(self,phi):
        try:
            square_root_term=    tf.math.sqrt(phi * (1 - phi))
            #square_root_term=tf.math.sqrt(tf.math.abs(phi) * tf.math.abs(1 - phi))
        except ValueError:
            raise ValueError("Cannot calculate the square root of a negative number")
        else:
            return np.pi/self.eta * square_root_term
    ###############################################
    #@tf.function
    def evaluate(self,X):

        lb = tf.reshape(self.lb, (1, -1))
        lb = tf.cast(self.lb, tf.float64)
        ub = tf.reshape(self.ub, (1, -1))
        ub = tf.cast(self.ub, tf.float64)

        H = (X - lb)/(ub - lb) 
        #tf.print("H.shape: ",H.shape)
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
    def test_IC(self,pathOutput):
        phi_pred = self.evaluate(self.X_u_test)
        phi_pred = np.reshape(phi_pred,(self.Nx,self.Ny,self.Nt))  
        #------------------------------------
        # plot
        num_boxes = 4   # Time intervals
        box_size = phi_pred.shape[2] // num_boxes

        # Compute the number of rows needed to display all subplots
        num_rows = (num_boxes + 1) // 2

        # Create the figure and subplots
        fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(8, 4*num_rows))

        # Loop over the boxes and plot the corresponding phi values

        for i, ax in enumerate(axes.flat):
            start_idx = i * box_size
            end_idx = (i + 1) * box_size
            im=ax.imshow(phi_pred[:, :, start_idx], cmap='jet')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            fig.colorbar(im, ax=ax)
            #ax.set_xlim([x.min(), x.max()])
            #ax.set_ylim([y.min(), y.max()])
            ax.set_title(f'Time: {start_idx}')

        filename = f"test_IC.jpg"
        plt.savefig(os.path.join(pathOutput ,filename))
        plt.close()
    ###############################################
    def loss_IC(self,x_ini,phi_ini):  
        #print("x_ini.shape: ",x_ini.shape)  
        #print("phi_ini.shape: ",phi_ini.shape)  
        phi_ini_pred=self.evaluate(x_ini)                                     
        loss_IC = tf.reduce_mean(tf.square(phi_ini-phi_ini_pred))
        
        """
        epoch=99
        if (epoch+1) % 100 == 0:
            plt.scatter(x_ini[:, 0], x_ini[:, 1],c=phi_ini)
            plt.show()
            plt.scatter(x_ini[:, 0], x_ini[:, 1],c=phi_ini_pred)
            plt.show()
        """   
        del phi_ini_pred, x_ini,phi_ini
        return loss_IC
    ###############################################
    def loss_BC(self,X_lb,X_ub,X_ltb,X_rtb,abs_x_min,abs_x_max,abs_y_min,abs_y_max):
        #tf.print("abs_x_min,abs_x_max,abs_y_min,abs_y_max: ", abs_x_min,abs_x_max,abs_y_min,abs_y_max)   
        #tf.print("X_ltb: ", X_ltb)
        #tf.print("X_ltb[:,1]: ", X_ltb[:,1])

        #X_ltb = tf.cast(X_ltb, dtype=tf.float64)
        #X_rtb = tf.cast(X_rtb, dtype=tf.float64)
        #X_ub = tf.cast(X_ub, dtype=tf.float64)
        #X_lb = tf.cast(X_lb, dtype=tf.float64)
  
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
    def loss_PDE(self, X_f):
        g = tf.Variable(X_f, dtype='float64', trainable=False)
        #tf.print("g: ", tf.shape(g))
        x_f = g[:, :2]  # x_f: x,y
        t = g[:, 2:3]
        x = g[:, 0:1]
        y = g[:, 1:2]
        #t=tf.convert_to_tensor(t, dtype=tf.float64)
        #x=tf.convert_to_tensor(x, dtype=tf.float64)
        #y=tf.convert_to_tensor(y, dtype=tf.float64)

        #tf.print("x,y,t shapes: ", tf.shape(x), tf.shape(y), tf.shape(t))
        #tf.print("x :",x)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(t)
            g = tf.concat([x,y, t], axis=1)
            phi = self.evaluate(g)
            #tf.print("phi.shape: ", tf.shape(phi) )
            #tf.print("phi: ", phi )
            tape.watch(phi)
            phi_x = tape.gradient(phi, x)
            #tf.print("phi_x ",phi_x)
            #tf.print("phi_x: ", tf.shape(phi_x))
            phi_y = tape.gradient(phi, y)
            phi_t = tape.gradient(phi, t)
            tape.watch(phi_x)
            phi_xx = tape.gradient(phi_x, x)
            phi_yy = tape.gradient(phi_y, y)
            lap_phi = phi_xx + phi_yy
        del tape

        #lap_phi = tf.convert_to_tensor(np.pi**2/(2*self.eta**2)*np.sin(np.pi*(r-self.R0)/self.eta)-np.pi/(r*self.eta)*np.cos(np.pi*(r-self.R0)/self.eta), dtype=tf.float64)
            
        phi_term = (np.pi**2 / (2 * self.eta**2)) * (2 * phi - 1)
        right_side_eqn = self.mu * ( self.sigma * ( lap_phi + phi_term)+ self.h(phi) * self.delta_g ) 
        f =phi_t -right_side_eqn 
        
        loss_f = tf.reduce_mean(tf.square(f))
        del phi_x,phi_y,phi_t,phi_xx,phi_yy,lap_phi,g, X_f, x_f,x,y,t,phi_term,right_side_eqn,f
        return loss_f
    ###############################################
    def loss(self,xf,x_ini,x_lb,x_ub,x_ltb,x_rtb,phi_ini,abs_x_min,abs_x_max,abs_y_min,abs_y_max):
        loss_IC = self.loss_IC(x_ini,phi_ini)      
        loss_f = self.loss_PDE(xf)        
        loss_BC = self.loss_BC(x_lb,x_ub,x_ltb,x_rtb,abs_x_min,abs_x_max,abs_y_min,abs_y_max)        
        loss =  self.f*loss_f +self.ic*loss_IC+self.bc*loss_BC #    
        del xf,x_ini,x_lb,x_ub,x_ltb,x_rtb,phi_ini
        return loss, loss_BC,loss_IC, loss_f # loss_BC,loss_IC, loss_f
    ###############################################
    def optimizerfunc(self,parameters):
        global list_loss_scipy
        global Nfeval
        self.set_weights(parameters)
               
        X_ini =self.X_ini_all_sub_domain[self.indices_ini]
        phi_ini=self.phi_ini_all_sub_domain[self.indices_ini]
        
        X_f = tf.convert_to_tensor(self.X_f_sub_domain_scipy, dtype=tf.float64)
        X_lb = tf.convert_to_tensor(self.X_lb_sub_domain, dtype=tf.float64)
        X_ub = tf.convert_to_tensor(self.X_ub_sub_domain, dtype=tf.float64)
        X_ltb = tf.convert_to_tensor(self.X_ltb_sub_domain, dtype=tf.float64)
        X_rtb = tf.convert_to_tensor(self.X_rtb_sub_domain, dtype=tf.float64)
        X_ini = tf.convert_to_tensor(X_ini, dtype=tf.float64)
        phi_ini = tf.convert_to_tensor(phi_ini, dtype=tf.float64)

        """
        if Nfeval==1:
          plt.scatter(X_ini[:, 0], X_ini[:, 1], cmap=plt.get_cmap('viridis'), c=phi_ini)
          plt.colorbar( shrink=0.35)
          plt.show()
        """
        #tf.print("X_f.shape:",X_f.shape)
        #tf.print("\n ")
        #tf.print("X_ini:",X_ini.shape)
        #tf.print("\n ")
        #tf.print("phi_ini:",phi_ini.shape)
        #tf.print("\n ")
        #tf.print("X_lb.shape:",X_lb.shape)
        #tf.print("\n ")
        #tf.print("X_ub.shape:",X_ub.shape)
        #tf.print("\n ")
        #tf.print("X_ltb.shape:",X_ltb.shape)
        #tf.print("\n ")
        #tf.print("X_rtb.shape:",X_rtb.shape)
        #tf.print("\n ")
        

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

        #self.set_weights(grads_1d.numpy())
        del grads, grads_w_1d,grads_b_1d
        return loss_val.numpy(), grads_1d.numpy()
    ###############################################
    def optimizer_callback(self, parameters):
        global Nfeval
        global list_loss_scipy

        if Nfeval % 50 == 0:  # Print during scipy iterations
            tf.print('Iter: {0:d}, total_loss: {1:.3e}, loss_BC: {2:.3e}, loss_IC: {3:.3e}, loss_f: {4:.3e}'.format(Nfeval, list_loss_scipy[-1][0], list_loss_scipy[-1][1], list_loss_scipy[-1][2], list_loss_scipy[-1][3]))
            #X_ini_ =self.X_ini_all_sub_domain[self.indices_ini]
            #phi_ini_=self.phi_ini_all_sub_domain[self.indices_ini]
            #X_ini_ = tf.convert_to_tensor(X_ini_, dtype=tf.float64)
            #phi_ini_ = tf.convert_to_tensor(phi_ini_, dtype=tf.float64)
            #tf.print("X_ini:",X_ini_.shape)
            #tf.print("\n ")
            #tf.print("phi_ini:",phi_ini_.shape)
            #tf.print("\n ")
            #del X_ini_,phi_ini_
        
        
        # Check if the loss is smaller than the threshold
        #if list_loss_scipy[-1][0] < self.thresh:
        #    tf.print("here")
        #    return True, list_loss_scipy  # Returning True stops the optimization

        Nfeval += 1
        
        return  list_loss_scipy  # Returning False continues the optimization
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
    def process_repository_files(self,path,pathOutput,title,filename):
        weights_files = self.PRE_POST.read_weights_files(path)
        #tf.print(weights_files)
        t_min = float('inf')
        t_max = float('-inf')

        # Make copy from self for testing and saving actual results
        PINN_= Sequentialmodel(layers=self.layers, X_f_train=self.X_f, X_ini_train=self.X_ini,\
                            X_lb_train=self.X_lb, X_ub_train=self.X_ub,\
                            X_ltb_train=self.X_ltb, X_rtb_train=self.X_rtb,\
                            phi_0=self.phi_0,phi_ini_train=self.phi_ini, N_ini=self.N_ini,X_u_test=self.X_u_test,\
                            X_ini_train_all=self.X_ini_train_all, phi_ini_train_all=self.phi_ini_train_all,\
                            X=None,T=None,x=self.x,y=self.y,lb=self.lb, ub=self.ub, mu=self.mu, sigma=self.sigma, delta_g=self.delta_g,\
                            R0=self.R0,X_center=self.X_center,Y_center=self.Y_center,eta=self.eta,\
                            Nx=self.Nx,Ny=self.Ny,Nt=self.Nt,phi_sol=None)
        
        phi_evolution = []
        for weights_file in weights_files:
            t_min, t_max = self.PRE_POST.extract_t_min_t_max(weights_file)
            weights_loaded = self.PRE_POST.load_weights(weights_file)
            PINN_.set_weights(weights_loaded)

            X_phi_test_sub = PINN_.X_u_test[:, 2] >= t_min
            X_phi_test_sub &= PINN_.X_u_test[:, 2] <= t_max
            X_phi_test_sub = PINN_.X_u_test[X_phi_test_sub, :]

            phi_pred = PINN_.evaluate(X_phi_test_sub).numpy()
            phi_pred = np.reshape(phi_pred, (PINN_.Nx, PINN_.Ny, -1))
            phi_evolution.append(phi_pred[:, :, 0])
        phi_evolution.append(phi_pred[:, :, -1])        
        num_boxes = 4     
        self.PRE_POST.plot_global_evolution(num_boxes,phi_evolution, pathOutput,title,filename,t_max)
        
        del PINN_,phi_evolution,X_phi_test_sub,phi_pred
    ###############################################
    def save_predictions(self,epoch,pathOutput,X_u_test,\
                                X_ini,u_ini,N_b): 
        title = f"φ predicted by PINN" # should be epoch+1 (this for debug purpose)
        filename = f"φ predicted by PINN.jpg"
        
        path_weights = "weights/"
        self.process_repository_files(path_weights,pathOutput,title,filename)
    ###############################################
    def save_predictions_regular_int(self,epoch,pathOutput,X_u_test,\
                                X_ini,u_ini,N_b,t_min, t_max):
        X_u_test_sub = X_u_test[:, 2] >= t_min
        X_u_test_sub &= X_u_test[:, 2] <= t_max
        X_u_test_sub = X_u_test[X_u_test_sub, :]
        phi_pred = self.evaluate(X_u_test_sub)
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

        thresh=1e-6

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
    def train(self,epochs,batch_size_max,N_batches,thresh,epoch_scipy_opt=1000,epoch_print=500, epoch_resample=100,\
              initial_check=False,save_reg_int=100,num_train_intervals=10,Nbr_pts_max_per_batch=1000,
              scipy_min_f_pts_per_batch=100,scipy_min_f_pts_per_batch_thresh=0.01,\
                  max_ic_scipy_pts=75,N_ini_min_per_batch=4  , ic_scipy_thresh=0.025, \
              discrete_resolv=True,path=None):  
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

        # loss
        list_loss=[]

        # dummy params
        flag=0
        flag_weights=1

        # shuffle Collocation points
        idx = np.random.permutation(self.X_f.shape[0])
        self.X_f = self.X_f[idx]
  
        # get N_b and N_ini
        N_b=self.X_lb.shape[0]
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
            for epoch in range(epochs):   
                if epoch>=0:
                    self.f=1
                    self.ic=1
                    self.bc=1      

                #tf.print("epoch: ", epoch)
                # set time bounds
                if discrete_resolv:
                    t_min, t_max = time_subdomains[count], time_subdomains[count+1] 
                else:
                    t_min, t_max = time_subdomains[0], time_subdomains[count+1] 
            
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
                if epoch==0 or flag:
                    X_f_sub_domain_scipy = []
                                        
                # **************** IC ***************
                if epoch==0 or flag:
                    if discrete_resolv==False:   # progressively increase IC points
                        num_elements_ini = int(len(X_ini) * (count+1) * (1/num_train_intervals) )  # to adjust the len of batch_X_ini by time interval
                    else:
                        num_elements_ini = int(len(X_ini) * (1/num_train_intervals) )
                                           
    
                    #**************************
                    num_x_intervals = int(np.ceil(np.sqrt(N_batches)))
                    num_y_intervals = int(np.ceil(np.sqrt(N_batches)))
                    x_interval_size = (X_lb_sub_domain[:, 0].max() - X_lb_sub_domain[:, 0].min()) / num_x_intervals
                    y_interval_size = (X_ltb_sub_domain[:, 1].max() - X_ltb_sub_domain[:, 1].min()) / num_y_intervals
                    
                    # Initialize the list to store the selected indices
                    selected_indices = []  # for Adam loss computing
                    # Iterate over the intervals and randomly select indices within each interval
                    for i in range(num_x_intervals):
                        for j in range(num_y_intervals):
                            # Define the x and y bounds for the current interval
                            x_lb = X_lb_sub_domain[:, 0].min() + i * x_interval_size
                            x_ub = X_lb_sub_domain[:, 0].min() + (i + 1) * x_interval_size
                            y_lb = X_ltb_sub_domain[:, 1].min() + j * y_interval_size
                            y_ub = X_ltb_sub_domain[:, 1].min() + (j + 1) * y_interval_size

                            # Find the indices of points within the current interval
                            interval_indices = np.where((X_ini_all[:, 0] >= x_lb) & (X_ini_all[:, 0] < x_ub) &
                                                        (X_ini_all[:, 1] >= y_lb) & (X_ini_all[:, 1] < y_ub))[0]

                            if len(interval_indices) > 0:
                                # Randomly select indices from the interval
                                selected_indices.extend(np.random.choice(interval_indices, size=min(len(interval_indices),int(num_elements_ini/N_batches) ), replace=False))
                    
                    # Select the corresponding initial points based on the selected indices
                    X_ini_sub_domain = X_ini_all[selected_indices]
                    phi_ini_sub_domain = phi_ini_all[selected_indices]
                    
                    #**************************
                    X_ini_all_sub_domain = X_ini_all #  All ini points (to use by the PINN if mini_batches are not filled)
                    phi_ini_all_sub_domain = phi_ini_all 

                    self.X_ini_sub_domain=X_ini_sub_domain
                    self.phi_ini_sub_domain = phi_ini_sub_domain

                    # move the IC points
                    if discrete_resolv and epoch>0:
                            X_ini_sub_domain[:,2]=t_min
                            X_ini_all_sub_domain[:,2]=t_min
                            phi_ini_all_sub_domain = self.evaluate(X_ini_all_sub_domain).numpy()
                            self.X_ini_sub_domain=X_ini_sub_domain   
                            phi_ini_sub_domain = self.evaluate(self.X_ini_sub_domain).numpy()
                            self.phi_ini_subdomain = phi_ini_sub_domain

                            self.X_ini_all_sub_domain=X_ini_all_sub_domain
                            self.phi_ini_all_sub_domain=phi_ini_all_sub_domain

                        
                        
                    # *********************************************************************
                    # --------------   prepare scipy IC points  --------------------------
                    # *********************************************************************
                    number_points_max = max_ic_scipy_pts  # Maximum points per batch for scipy (computing)
                    percentage_threshold = ic_scipy_thresh # Percentage threshold for determining coef_points
                    selected_indices_scipy = []  # IC indices for scipy optimizer
                    
                    # Iterate over the intervals and randomly select indices within each interval
                    for i in range(num_x_intervals):
                        for j in range(num_y_intervals):
                            # Define the x and y bounds for the current interval
                            x_lb = X_lb_sub_domain[:, 0].min() + i * x_interval_size
                            x_ub = X_lb_sub_domain[:, 0].min() + (i + 1) * x_interval_size
                            y_lb = X_ltb_sub_domain[:, 1].min() + j * y_interval_size
                            y_ub = X_ltb_sub_domain[:, 1].min() + (j + 1) * y_interval_size

                            # Find the indices of points within the current interval
                            interval_indices = np.where((self.X_ini_all_sub_domain[:, 0] >= x_lb) & (self.X_ini_all_sub_domain[:, 0] < x_ub) &
                                                        (self.X_ini_all_sub_domain[:, 1] >= y_lb) & (self.X_ini_all_sub_domain[:, 1] < y_ub))[0]
                            total_interface_points = np.sum((self.phi_ini_all_sub_domain[interval_indices] >5e-2) & (self.phi_ini_all_sub_domain[interval_indices] <1 ))
                            percentage_interface_points = total_interface_points / len(self.phi_ini_all_sub_domain[interval_indices])
                            #number_points_for_scipy = min(int(percentage_interface_points / percentage_threshold), number_points_max)
                            number_points_for_scipy = min(int(percentage_interface_points *number_points_max), number_points_max)
                            if len(interval_indices) > 0:   # number of points per batch
                                selected_indices_scipy.extend(np.random.choice(interval_indices, size=max(number_points_for_scipy,N_ini_min_per_batch), replace=True))

                    self.indices_ini =selected_indices_scipy # update indices for scipy optimization         
                    #tf.print(t_min,t_max, "self.indices_ini", self.indices_ini)
                    plt.scatter(self.X_ini_all_sub_domain[self.indices_ini][:, 0], self.X_ini_all_sub_domain[self.indices_ini][:, 1],s=0.5, cmap=plt.get_cmap('jet'), c=self.phi_ini_all_sub_domain[self.indices_ini])
                    plt.colorbar( shrink=0.35)
                    title=f"Scipy optimizer IC points at Epoch {epoch} for Time interval: t_min: {t_min:.3f}, t_max: {t_max:.3f}.jpg"
                    phi_ini_length = len(self.phi_ini_all_sub_domain[self.indices_ini])
                    plt.title(f'Number of IC points for Scipy optimization: {phi_ini_length}',fontsize=8)
                    plt.grid(True)
                    plt.xticks(np.linspace(X_lb_sub_domain[:, 0].min(), X_ub_sub_domain[:, 0].max(), num_x_intervals+1))
                    plt.yticks(np.linspace(X_ltb_sub_domain[:, 1].min(),X_ltb_sub_domain[:, 1].max(),num_y_intervals+1))
                    plt.savefig(os.path.join(path,title), dpi = 500, bbox_inches='tight')
                    plt.close()            
                    
                    X_ini_sub_domain = X_ini_all[selected_indices_scipy]
                    phi_ini_sub_domain = phi_ini_all[selected_indices_scipy] 
                    # *********************************************
                    # check initial condition (epoch==0 or domain change)
                    if initial_check:
                        self.plot_ini(X_ini_sub_domain,phi_ini_sub_domain,X_ini,self.Nx,self.Ny,path,t_min,t_max, epoch)
                    """
                    # Set x-axis and y-axis limits
                    plt.xlim(0, 1)
                    plt.ylim(0, 1)

                    # Scatter plot
                    plt.scatter(self.X_ini_sub_domain[:, 0], self.X_ini_sub_domain[:, 1], cmap=plt.get_cmap('viridis'), c=self.phi_ini_sub_domain)
                    plt.colorbar( shrink=0.35)

                    # Show the plot
                    plt.show()

                    plt.scatter(X_ini_all_sub_domain[:, 0], X_ini_all_sub_domain[:, 1], cmap=plt.get_cmap('viridis'), c=phi_ini_all_sub_domain)
                    plt.colorbar( shrink=0.35)

                    # Show the plot
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
                all_batches_X_ini =[] 
                all_batches_phi_ini = []
                all_batches_X_lb = []
                all_batches_X_ub = []
                all_batches_X_ltb = []
                all_batches_X_rtb = []
                all_batches_X_ini_all =[] # All ini points (to use by the PINN if mini_batches are not filled)
                all_batches_phi_ini_all = [] # All ini points (to use by the PINN if mini_batches are not filled)

                # Loop over the intervals and group the points based on their location  
                if (epoch==0) or flag:     
                    fig, ax = plt.subplots()          
                for i in range(num_x_intervals):
                    for j in range(num_y_intervals):
                        # Define the boundaries for the subdomain
                        x_min = X_lb_sub_domain[:, 0].min() + i * x_interval_size
                        x_max = X_ub_sub_domain[:, 0].min() + (i + 1) * x_interval_size
                        y_min = X_ltb_sub_domain[:, 1].min() + j * y_interval_size
                        y_max = X_rtb_sub_domain[:, 1].min() + (j + 1) * y_interval_size
                        #tf.print(x_min,x_max,y_min,y_max)
                        
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
                            (X_ini_all_sub_domain[:, 0] >= x_min) &
                            (X_ini_all_sub_domain[:, 0] <= x_max) &
                            (X_ini_all_sub_domain[:, 1] >= y_min) &
                            (X_ini_all_sub_domain[:, 1] <=
                              y_max)
                        )[0].astype(int)
                        bacth_X_ini_all_indices=np.asarray(bacth_X_ini_all_indices)
                        batch_X_ini_all = X_ini_all_sub_domain[bacth_X_ini_all_indices]
                        batch_phi_ini_all = phi_ini_all_sub_domain[bacth_X_ini_all_indices]
                        # *********************************************************************
                        # ******************     Adaptive minibatching   Scipy  ***************
                        # *********************************************************************
                        ### new                         
                        number_points_min_per_batch = scipy_min_f_pts_per_batch   # can be adjustable
                        percentage_threshold = scipy_min_f_pts_per_batch_thresh        # can be adjustable
                        total_interface_points_per_batch = np.sum((batch_phi_ini_all>5e-2) & (batch_phi_ini_all <1))
                        percentage_interface_points_per_batch = total_interface_points_per_batch / len(batch_phi_ini_all)
                        number_points_per_batch = max(int(percentage_interface_points_per_batch / percentage_threshold)*number_points_max, number_points_min_per_batch)
                        #number_points_per_batch=min(number_points_per_batch,Nbr_pts_max_per_batch)
                        number_points_per_batch=max(number_points_per_batch,int(percentage_interface_points_per_batch*Nbr_pts_max_per_batch))
    
                        if len(bacth_X_ini_all_indices) > 0:
                            new_indices_collocation_pts=np.random.choice(len(batch_Xf), size=number_points_per_batch, replace=True)
                        batch_Xf=batch_Xf[new_indices_collocation_pts]
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
                        all_batches_X_ini.append(batch_X_ini)
                        all_batches_phi_ini.append(batch_phi_ini)
                        all_batches_X_lb.append(batch_X_lb)
                        all_batches_X_ub.append(batch_X_ub)
                        all_batches_X_ltb.append(batch_X_ltb)
                        all_batches_X_rtb.append(batch_X_rtb)
                        all_batches_X_ini_all.append(batch_X_ini_all)
                        all_batches_phi_ini_all.append(batch_phi_ini_all)
                                              
                        # plot the subdomain
                        if (epoch==0) or flag:    
                            color = np.random.rand(3)
                            ax.scatter(batch_Xf[:, 0], batch_Xf[:, 1], color=color, marker='*',s=0.5, label='PDE Collocation')
                            ax.scatter(batch_X_ini[:, 0], batch_X_ini[:, 1], color=color, marker='o',s=25, label='IC')
                            ax.scatter(batch_X_lb[:, 0], batch_X_lb[:, 1], color=color, marker='v',s=5, label='Lower Boundary')
                            ax.scatter(batch_X_ub[:, 0], batch_X_ub[:, 1], color=color, marker='^',s=5, label='Upper Boundary')
                            ax.scatter(batch_X_ltb[:, 0], batch_X_ltb[:, 1], color=color, marker='<',s=5, label='Left Boundary')
                            ax.scatter(batch_X_rtb[:, 0], batch_X_rtb[:, 1], color=color, marker='>',s=5, label='Right Boundary')
                
                all_batches_Xf = np.asarray(all_batches_Xf, dtype=object)       
                all_batches_X_lb = np.asarray(all_batches_X_lb, dtype=object)    
                all_batches_X_ub = np.asarray(all_batches_X_ub, dtype=object)
                all_batches_X_ltb = np.asarray(all_batches_X_ltb, dtype=object)    
                all_batches_X_rtb = np.asarray(all_batches_X_rtb, dtype=object)           
                all_batches_X_ini=np.asarray(all_batches_X_ini, dtype=object)        
                all_batches_phi_ini=np.asarray(all_batches_phi_ini, dtype=object)

                # Concatenate all the tensors 
                all_batches_X_ini_to_plot = all_batches_X_ini[0]
                all_batches_phi_ini_to_plot = np.reshape(all_batches_phi_ini[0], (1, -1))

                for i in range(1, len(all_batches_X_ini)):
                    all_batches_X_ini_to_plot = np.vstack((all_batches_X_ini_to_plot, all_batches_X_ini[i]))
                    phi_ini_reshaped = np.reshape(all_batches_phi_ini[i], (1, -1))
                    all_batches_phi_ini_to_plot = np.hstack((all_batches_phi_ini_to_plot, phi_ini_reshaped))

                if (epoch==0) or flag:  
                    scatter_ini=ax.scatter(all_batches_X_ini_to_plot[:, 0], all_batches_X_ini_to_plot[:, 1],cmap=plt.get_cmap('viridis') ,c=all_batches_phi_ini_to_plot,marker='o',s=10, label="IC")  
                    cbar = plt.colorbar(scatter_ini, ax=ax, shrink=0.35, label=r"$\phi$")
                    title = f"Adam Subdomains\nEpoch {epoch}, {int(num_x_intervals*num_x_intervals)} minibatches\n(t_min = {t_min:.3f}, t_max = {t_max:.3f})" 
                    plt.title(title, fontsize=10)
                    if flag_filled:
                        plt.text(0.5, 0.5, "IC points well filled", color="orange", fontsize=12, ha="center", va="center")
                    else: 
                        plt.text(0.5, 0.5, "!!! IC points auto-filled",color="red" , fontsize=12, ha="center", va="center")
                    #plt.legend()
                    plt.savefig(os.path.join(path, f"Adam_Subdomains_Epoch_{epoch}_{int(num_x_intervals*num_x_intervals)}_minibatches_t_min_{t_min:.3f}_t_max_{t_max:.3f}.jpg"), dpi=500, bbox_inches='tight')
                    plt.close() 
                    ####################################################################

                global_loss = 0.0
                global_loss_f = 0.0
                global_loss_IC = 0.0
                global_loss_BC = 0.0
                ######################################################################################
                # Main loop for minibatching : compute loss for each batch  ##########################
                ######################################################################################
                for batch_idx, (batch_X_f, batch_X_ini,batch_X_ini_all,batch_phi_ini,batch_phi_ini_all,\
                                 batch_X_lb, batch_X_ub, batch_X_ltb, batch_X_rtb)\
                                      in enumerate(zip(all_batches_Xf, all_batches_X_ini,all_batches_X_ini_all,\
                                                        all_batches_phi_ini,all_batches_phi_ini_all, all_batches_X_lb,\
                                                              all_batches_X_ub, all_batches_X_ltb, all_batches_X_rtb)):

                    if len(batch_X_ini)==0:  # fill this batch with IC points and corresponding phi 
                        idx_ini_all = np.random.choice(batch_X_ini_all.shape[0], N_ini_min_per_batch, replace=False)
                        batch_X_ini=batch_X_ini_all[idx_ini_all]
                        batch_phi_ini=batch_phi_ini_all[idx_ini_all]
                        batch_phi_ini = batch_phi_ini.reshape(-1, 1)
                        self.X_ini_sub_domain = tf.concat([self.X_ini_sub_domain, batch_X_ini], axis=0)
                        self.phi_ini_sub_domain = tf.concat([self.phi_ini_sub_domain, batch_phi_ini], axis=0)

                    if (epoch==0 or flag) and (batch_idx==0):
                        self.PRE_POST.plot_domain(X_ini_sub_domain,batch_X_ini,X_f_sub_domain,batch_X_f,X_ub_sub_domain,batch_X_ub,\
                                                    X_lb_sub_domain,batch_X_lb,X_ltb_sub_domain,batch_X_ltb,X_rtb_sub_domain,batch_X_rtb,\
                                                    t_min, t_max,epoch,batch_idx,phi_0=self.phi_0,phi_ini_train=phi_ini_sub_domain,phi_ini_train_s=batch_phi_ini,path=path)  
                        

                    batch_X_f = tf.convert_to_tensor(batch_X_f, dtype=tf.float64)
                    batch_X_ini = tf.convert_to_tensor(batch_X_ini, dtype=tf.float64)
                    batch_X_lb = tf.convert_to_tensor(batch_X_lb, dtype=tf.float64)
                    batch_X_ub = tf.convert_to_tensor(batch_X_ub, dtype=tf.float64)
                    batch_X_ltb = tf.convert_to_tensor(batch_X_ltb, dtype=tf.float64)
                    batch_X_rtb = tf.convert_to_tensor(batch_X_rtb, dtype=tf.float64)
                    batch_phi_ini = tf.convert_to_tensor(batch_phi_ini, dtype=tf.float64)

                    with tf.GradientTape() as tape:
                        loss, loss_BC,loss_IC, loss_f = self.loss(batch_X_f,batch_X_ini,\
                                                                batch_X_lb,batch_X_ub,batch_X_ltb,batch_X_rtb,\
                                                                    batch_phi_ini,self.abs_x_min,self.abs_x_max,self.abs_y_min,self.abs_y_max)
                        
                    
                    gradients = tape.gradient(loss, self.trainable_variables)
                    results=self.optimizer_Adam.apply_gradients(zip(gradients, self.trainable_variables))
                    del tape 
                    
                    global_loss += loss
                    global_loss_f += loss_f
                    global_loss_IC += loss_IC
                    global_loss_BC+= loss_BC
                    list_loss.append([global_loss,global_loss_BC,global_loss_IC,global_loss_f])                    
                    #tf.print(epoch, len(self.trainable_variables))                                                                      
                    # Update Scipy Collocation points list during minibatching    
                    if (epoch==0 or flag):    
                        for batch_Xf_row in zip(batch_X_f):                           
                            X_f_sub_domain_scipy.append(batch_Xf_row)   
                       
                if (epoch==0 or flag):        
                    # Get and check Scipy Collocation points list         
                    X_f_sub_domain_scipy=np.asarray(X_f_sub_domain_scipy, dtype=np.float64)
                    X_f_sub_domain_scipy = np.reshape(X_f_sub_domain_scipy, (len(X_f_sub_domain_scipy), 3))
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
                    
                    flag=0  # very important : all data are now prepared for minibatching ==> drop the flag 
                    del X_f_sub_domain_scipy
                    
                # Delete variables to free up memory
                del all_batches_Xf, all_batches_X_ini, all_batches_phi_ini, all_batches_X_lb, all_batches_X_ub, all_batches_X_ltb, \
                    all_batches_X_rtb, all_batches_X_ini_all, all_batches_phi_ini_all, gradients,batch_X_f,batch_X_ini,\
                                                                batch_X_lb,batch_X_ub,batch_X_ltb,batch_X_rtb,\
                                                                    batch_phi_ini, new_indices_collocation_pts                                                 
                    
                ####################################################################################################
                ####################################################################################################
                ###############  End Minibatching   ################################################################
                ####################################################################################################
                ####################################################################################################              
                # Save predictions at regular time intervals
                if (epoch+1)  % save_reg_int== 0:  
                    self.save_predictions_regular_int(epoch,path,self.X_u_test,\
                                X_ini,phi_ini,N_b,t_min, t_max)
                
                # Compute the average loss for the epoch
                global_loss /= (num_x_intervals * num_y_intervals)
                global_loss_f /= (num_x_intervals * num_y_intervals)
                global_loss_IC /= (num_x_intervals * num_y_intervals)
                global_loss_BC/= (num_x_intervals * num_y_intervals)

                # print losses
                if epoch % epoch_print == 0:    
                    tf.print('Epoch: {0:d}, total_loss: {1:.3e}, loss_BC: {2:.3e}, loss_IC: {3:.3e}, loss_f: {4:.3e}'.format(epoch, global_loss, global_loss_BC,global_loss_IC, global_loss_f))
                    tf.print("\n ")
                
                ##################################################
                ##################################################
                ###############  Scipy Optimizer   ##############
                ##################################################
                ##################################################   
                # call scipy optimizer if loss > thresh
                if epoch % epoch_scipy_opt == 0 and global_loss > self.thresh and epoch>0: 
                    global Nfeval
                    global list_loss_scipy
                    list_loss_scipy = []
                    init_params = self.get_weights().numpy()
                    tf.print("\n")
                    tf.print("!!! Scipy optimize: !!! - Epoch: ",str(epoch))
                    Nfeval=1  # reinitialize the global value of Nfeval
                    
                    results = scipy.optimize.minimize(fun = self.optimizerfunc, 
                                                    x0 = init_params, 
                                                    args=(), 
                                                    method= 'L-BFGS-B', 
                                                    jac= True,        # If jac is True, fun is assumed to return the gradient along with the objective function
                                                    callback = self.optimizer_callback, 
                                                    options = {'disp': None,
                                                                'maxiter': 50000,     
                                                                'iprint': -1})
                    #self.save_predictions(epoch,path,self.X_u_test,X_ini,phi_ini,N_b,t_min, t_max)
                    self.set_weights(results.x)

                    global_loss,global_loss_BC,global_loss_IC, global_loss_f =list_loss_scipy[-1]
 
                    #list_loss.append([loss,loss_BC, loss_IC,loss_f]) # to reactivate
                    tf.print('==> loss after L-BFGS-B optimization for Epoch: {0:d}, total_loss: {1:.3e}, loss_BC: {2:.3e}, loss_IC: {3:.3e}, loss_f: {4:.3e}'.format(epoch, global_loss,global_loss_BC,global_loss_IC, global_loss_f))
                    tf.print("!!! Scipy optimization done !!!\n ")
                    del results,list_loss_scipy,init_params
                ##################################################
                ##################################################
                ###############  Save and chane Domain ###########
                ##################################################
                ##################################################                   
                # save weights and train on the next time domain
                if global_loss < self.thresh and t_max<=1 and flag==0: #and epoch>0 and epoch % 10==0  : #the last is added just to ensure that the model train a bit before new changes (continuity)
                    if  flag_weights:  # save weights at each time-domain change 

                        # save weights
                        weights_file = 'weights/weights_tmin_{:.3f}_tmax_{:.3f}_{}.json'.format(t_min, t_max, weights_key)
                        weights_dict[weights_key] = {}
                        weights_dict[weights_key]['t_min'] = t_min
                        weights_dict[weights_key]['t_max'] = t_max
                        weights_dict[weights_key]['weights'] = [w.numpy() for w in self.get_weights()]
                        with open(weights_file, 'w') as f:
                            json.dump(weights_dict[weights_key], f)

                        # save predictions 
                        self.save_predictions(epoch,path,self.X_u_test,\
                                    X_ini,phi_ini,N_b,t_min, t_max)

                        if t_max==1:   # stop saving weigthts (alles abgeschlossen)
                            self.thresh/=1.1
                            tf.print("Now optimizing the solution for the new threshold: {:.3e}".format(self.thresh))
                            #flag_weights=0                            

                    # prepare training on the next time domain
                    if t_max<1:
                        tf.print('Increase time interval ==> Epoch: {0:d}, total_loss: {1:.3e}, loss_BC: {2:.3e}, loss_IC: {3:.3e}, loss_f: {4:.3e}'.format(epoch, global_loss, global_loss_BC,global_loss_IC, global_loss_f))
                        tf.print("\n ")
                        count+=1
                        # set new time bounds
                        if discrete_resolv:
                            t_min, t_max =time_subdomains[count], time_subdomains[count+1] 
                        else:
                            t_min, t_max =time_subdomains[0], time_subdomains[count+1] 
                    
                        tf.print("change the time domain to: ",'t_min: {0:.3f}, t_max: {1:.3f}'.format(t_min, t_max))
                        # increase batch size ==> raise the flag 
                        flag=1    

        f_mem.close()
        return list_loss
    ###############################################
