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
np.random.seed(1234)
tf.random.set_seed(1234)
import random
import datetime
import shutil
import random

from importlib import reload
import pre_post
reload(pre_post) 
from pre_post import *



class Sequentialmodel(tf.Module):
    ###############################################
    def __init__(self, layers, X_f_train, X_ini_train, X_lb_train, X_ub_train,\
                  phi_ini_train, X_u_test, X,T,x,lb, ub, mu,\
                      sigma, delta_g, eta, name=None):
        super().__init__(name=name)

        self.X_f = X_f_train             
        self.X_ini = X_ini_train
        self.X_lb = X_lb_train
        self.X_ub = X_ub_train
        self.phi_ini = phi_ini_train
        self.X_u_test = X_u_test
        self.lb = lb
        self.ub = ub
        self.mu = mu
        self.sigma = sigma
        self.delta_g = delta_g
        self.eta = eta
        self.layers = layers

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
        self.optimizer_Adam = tf.keras.optimizers.Adam(learning_rate=0.001) 
 
        self.PRE_POST=PrePost(X=X, T=T, lb=lb, ub=ub, x=x, eta=eta)
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
    """
    def evaluate(self,x):
        
        x = (x-self.lb)/(self.ub-self.lb)
        
        a = x
        
        for i in range(len(self.layers)-2):
            
            z = tf.add(tf.matmul(a, self.W[2*i]), self.W[2*i+1])
            a = tf.nn.relu(z)   
        a = tf.add(tf.matmul(a, self.W[-2]), self.W[-2]) # For regression, no activation to last layer
        a = tf.nn.relu(a)  
        #a = tf.math.sigmoid(tf.add(tf.matmul(a, self.W[-2]), self.W[-1]))
        return a
    """
    def evaluate(self,X):
        H = (X - self.lb)/(self.ub - self.lb) 
        #print(np.asarray(H).min(),np.asarray(H).max())
        #print(np.asarray(self.W).shape)
        #print(np.asarray(H).shape)

        for l in range(0,len(self.layers)-2):
            W = self.W[2*l]
            b = self.W[2*l+1]
            H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))

        W = self.W[-2]
        b = self.W[-1]
        Y = tf.math.add(tf.matmul(H, W), b) # For regression, no activation to last layer
        Y = tf.nn.sigmoid(Y) # apply sigmoid activation function
        return Y
    ###############################################
    def get_weights(self):

        parameters_1d = []  # [.... W_i,b_i.....  ] 1d array
        
        for i in range (len(self.layers)-1):
            
            w_1d = tf.reshape(self.W[2*i],[-1])   #flatten weights 
            b_1d = tf.reshape(self.W[2*i+1],[-1]) #flatten biases
            
            parameters_1d = tf.concat([parameters_1d, w_1d], 0) #concat weights 
            parameters_1d = tf.concat([parameters_1d, b_1d], 0) #concat biases
        
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
            parameters = np.delete(parameters,np.arange(size_b),0) #delete 
    ###############################################
    def loss_IC(self,x_ini,phi_ini):                                         
        loss_IC = tf.reduce_mean(tf.square(phi_ini-self.evaluate(x_ini)))
        return loss_IC
    ###############################################
    def loss_BC(self,x_lb,x_ub):   
        pred_lb=self.evaluate(x_lb)
        pred_ub=self.evaluate(x_ub)
        loss_phi =  tf.reduce_mean(tf.square(pred_lb-1))+ tf.reduce_mean(tf.square(pred_ub)) #+tf.reduce_mean(tf.square(pred_ub+1)) 
        return loss_phi
    ###############################################
    def loss_PDE(self, X_f):
        g = tf.Variable(X_f, dtype = 'float64', trainable = False)

        x_f = g[:,0:1]
        t_f = g[:,1:2]
        #print("x_f.shape,t_f.shape :",t_f.shape,t_f.shape)

        with tf.GradientTape(persistent=True) as tape:

            tape.watch(x_f)
            tape.watch(t_f)

            g = tf.stack([x_f[:,0], t_f[:,0]], axis=1) 

            phi = self.evaluate(g)
            #phi =tf.clip_by_value(phi, clip_value_min=0, clip_value_max=tf.reduce_max(phi) )
            #print("phi.min(),phi.max(): ",tf.reduce_min(phi),tf.reduce_max(phi) )
            #phi = tf.round(phi * 1e6) / 1e6  # round phi to 6 decimal places
            phi_x = tape.gradient(phi,x_f)

            phi_t = tape.gradient(phi,t_f)    
            phi_xx = tape.gradient(phi_x, x_f)

        del tape

        lap_phi = phi_xx 
        phi_term = (np.pi**2 / (2 * self.eta**2)) * (2 * phi - 1)
        right_side_eqn = self.mu * (self.sigma * (lap_phi + phi_term) + self.h(phi) * self.delta_g)
        f =phi_t -right_side_eqn 

        loss_f = tf.reduce_mean(tf.square(f))
        
        return loss_f
    ###############################################
    def loss(self,xf,x_ini,x_lb,x_ub,phi_ini):
        
        loss_IC = self.loss_IC(x_ini,phi_ini)      
        
        loss_f =self.loss_PDE(xf) 
        loss_BC = self.loss_BC(x_lb,x_ub) 
        loss = loss_IC + loss_BC+ loss_f
        
        return loss, loss_BC,loss_IC, loss_f
    ###############################################
    def optimizerfunc(self,parameters):
        
        self.set_weights(parameters)
       
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            
            loss_val, loss_BC,loss_IC, loss_f = self.loss(self.X_f,self.X_ini,self.X_lb,self.X_ub,self.phi_ini)   
        grads = tape.gradient(loss_val,self.trainable_variables)
                
        del tape
        
        grads_1d = [ ] #flatten grads 
        
        for i in range (len(self.layers)-1):

            grads_w_1d = tf.reshape(grads[2*i],[-1]) #flatten weights 
            grads_b_1d = tf.reshape(grads[2*i+1],[-1]) #flatten biases

            grads_1d = tf.concat([grads_1d, grads_w_1d], 0) #concat grad_weights 
            grads_1d = tf.concat([grads_1d, grads_b_1d], 0) #concat grad_biases

        return loss_val.numpy(), grads_1d.numpy()
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
    def save_predictions(self,epoch,pathOutput,X_u_test,\
                                X_ini,u_ini,N_b,t_min, t_max):
        u_pred = self.evaluate(X_u_test)
        dim=u_pred.shape[0]// 100
        u_pred = np.reshape(u_pred,(dim,100),order='F')  # 10 time intervals
        #------------------------------------
        # plot
        # for real time control (show predictions)
        plt.figure() 
        n_intervals = 4
        t_step = len(u_pred.T) // n_intervals

        plt.plot(X_ini[:,0],u_ini[:,0],'b--', linewidth = 2, label = "u_ini_true")
        
        for i in range(n_intervals):
            color_ =['c','m','g','orange']
            t_idx = i * t_step
            x_p=np.linspace(-1,1,len(u_pred.T[t_idx, :]))
            label = 't_{}'.format(t_idx)+"_pred"
            plt.plot(x_p, u_pred.T[t_idx, :], '--', linewidth=2,color=color_[i], label=label)            
        color_ = tuple(np.random.rand(3))
        plt.plot(x_p, u_pred.T[-1, :], 'r--', linewidth=2, label='t_f')
        plt.legend()                            
        filename="u_pred_epoch_"+str(epoch+1)+' - t_min: {0:.3f}, t_max: {1:.3f}'.format(t_min, t_max)+".jpg"
        plt.savefig(os.path.join(pathOutput ,filename))
        plt.close()
    ###############################################
    def plot_ini(self,batch_X_ini,batch_u_ini,X_ini,u_ini,path,epoch):
        #print(batch_X_ini.shape,batch_u_ini.shape)
        plt.plot( batch_X_ini[:,0], batch_u_ini,'o')
        filename = f"IC_epoch{epoch}.png"
        plt.savefig(os.path.join(path, filename))
        plt.close()
    ###############################################
    def optimizer_callback(self,parameters):
            global Nfeval
            global list_loss_scipy
            list_loss_scipy=[]
                        
            total_loss,loss_BC,loss_IC, loss_f = self.loss(self.X_f,self.X_ini,self.X_lb,self.X_ub,self.phi_ini)  
            
            #u_pred = self.evaluate(X_u_test)
            #error_vec = np.linalg.norm((u-u_pred),2)/np.linalg.norm(u,2)
            if Nfeval % 1000 == 0:       # how much to print during scipy iterations
                    tf.print('Iter: {0:d}, total_loss: {1:.3e}, loss_BC: {2:.3e}, loss_IC: {3:.3e}, loss_f: {4:.3e}'.format(Nfeval, total_loss, loss_BC,loss_IC, loss_f))
            list_loss_scipy.append([total_loss,loss_BC,loss_IC, loss_f])
            Nfeval += 1
            return list_loss_scipy
   ###############################################
    ###############################################
    def train(self,epochs,batch_size,thresh,epoch_scipy_opt=1000,epoch_print=500, resample=False,\
                initial_check=False,save_reg_int=False,num_train_intervals=10,moving_IC=True,path=None):  

        # time intervals 

        time_subdomains=np.linspace(0,1,num_train_intervals+1)
        count=0
        # init
        X_ini=self.X_ini
        phi_ini=self.phi_ini
        # loss
        global Nfeval
        list_loss=[]
        # dummy params
        flag=0

        # shuffle Collocation points
        idx = np.random.permutation(self.X_f.shape[0])
        self.X_f = self.X_f[idx]

        # get N_b and N_ini
        N_b=self.X_lb.shape[0]
        N_ini=self.X_ini.shape[0]

        ############################################
        for epoch in range(epochs):

            # set time bounds
            if moving_IC:
                t_min, t_max =time_subdomains[count], time_subdomains[count+1] 
            else:
                t_min, t_max =time_subdomains[0], time_subdomains[count+1] 
            
            if resample:
                # re-shuffle training points for each iteration
                idx = np.random.permutation(self.X_f.shape[0])
                self.X_f = self.X_f[idx]

            # update selection from Collocation points
            X_f = self.X_f[np.logical_and(t_min <= self.X_f[:,1], self.X_f[:,1] <= t_max)]
            idx_b = np.random.permutation(X_f.shape[0])

            # update selection from bouńdaries   
            batch_X_lb = self.X_lb[np.logical_and(t_min <= self.X_lb[:,1], self.X_lb[:,1] <= t_max)]
            batch_X_ub = self.X_ub[np.logical_and(t_min <= self.X_ub[:,1], self.X_ub[:,1] <= t_max)]

            num_elements_ini = int(len(X_ini)  )  # to adjust the len of batch_X_ini
            indices_ini = sorted(random.sample(range(len(X_ini)), num_elements_ini ))
            batch_X_ini = X_ini[indices_ini]

            # move the IC points
            if moving_IC:
                    batch_X_ini[:,1]=count * (1/num_train_intervals)

            # new IC from model prediction
            if count==0:
                batch_phi_ini= phi_ini[indices_ini]
            else:
                if moving_IC:
                        batch_phi_ini = self.evaluate(batch_X_ini) # moving IC

            # check initial condition
            if initial_check:
                if epoch==0 or flag:
                    self.plot_ini(batch_X_ini,batch_phi_ini,X_ini,batch_phi_ini,path,epoch)
            
            ###############  Minibatching   ##################  
            # # only on X_f,  X_lb, X_ub and    
            for batch_idx in range(0, X_f.shape[0], batch_size):
                batch_X_f = X_f[idx_b[batch_idx:batch_idx+batch_size]]
                #t_batch = np.random.uniform(t_min, t_max, batch_X_f.shape[0])
                #batch_X_f[:,1]=t_batch

                if (epoch==0 or flag) and (batch_idx==0):
                    self.PRE_POST.plot_domain(batch_X_ini,X_f,batch_X_ub,batch_X_lb,t_min, t_max,batch_idx=0)  
                    flag=0  
                
                with tf.GradientTape() as tape:
                    loss, loss_BC,loss_IC, loss_f = self.loss(batch_X_f,batch_X_ini,batch_X_lb,batch_X_ub,batch_phi_ini)
                    list_loss.append([loss,loss_BC, loss_IC,loss_f])
                gradients = tape.gradient(loss, self.trainable_variables)
                results=self.optimizer_Adam.apply_gradients(zip(gradients, self.trainable_variables))
            
            # Save predictions at regular time intervals
            if save_reg_int and (epoch+1) % 1000 == 0: 
                #batch_X_u_test=X_u_test[np.logical_and(t_min <= X_u_test[:,1], X_u_test[:,1] <= t_max)]
                self.save_predictions(epoch,path,self.X_u_test,\
                            X_ini,phi_ini,N_b,t_min, t_max)

            # print losses
            if epoch % epoch_print == 0:    
                tf.print('Epoch: {0:d}, total_loss: {1:.3e}, loss_BC: {2:.3e}, loss_IC: {3:.3e}, loss_f: {4:.3e}'.format(epoch, loss, loss_BC,loss_IC, loss_f))
            
            # call scipy optimizer if loss > thresh
            if epoch % epoch_scipy_opt == 0 and epoch>0 and loss > thresh: 
                init_params = self.get_weights().numpy()
                print("\n")
                print("!!! Scipy optimize: !!! - Epoch: ",str(epoch))
                Nfeval=1  # reinitialize the global value of Nfeval
                
                results = scipy.optimize.minimize(fun = self.optimizerfunc, 
                                                x0 = init_params, 
                                                args=(), 
                                                method='L-BFGS-B', 
                                                jac= True,        # If jac is True, fun is assumed to return the gradient along with the objective function
                                                callback = self.optimizer_callback, 
                                                options = {'disp': None,
                                                            'maxcor': 200, 
                                                            'ftol': 1 * np.finfo(float).eps,  #The iteration stops when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol
                                                            'gtol': 5e-8, 
                                                            'maxfun':  1000, 
                                                            'maxiter': 1000,     
                                                            'iprint': -1,   #print update every N iterations
                                                            'maxls': 50})
                #print(results.x)
                self.set_weights(results.x)

                loss,loss_BC,loss_IC, loss_f =list_loss_scipy[-1]
                list_loss.append([loss,loss_BC, loss_IC,loss_f])
                tf.print('==> loss after L-BFGS-B optimization for Epoch: {0:d}, total_loss: {1:.3e}, loss_BC: {2:.3e}, loss_IC: {3:.3e}, loss_f: {4:.3e}'.format(epoch, loss, loss_BC,loss_IC, loss_f))
                print("!!! Scipy optimization done !!!\n ")

            # save weights and train on the next time domain
            if loss < thresh and t_max<1  and epoch % 100==0: # is added just to ensure that the model train  a bit before new changes
                self.save_predictions(epoch,path,self.X_u_test,\
                            X_ini,phi_ini,N_b,t_min, t_max)
                count+=1
                
                # set time bounds
                if moving_IC:
                    t_min, t_max =time_subdomains[count], time_subdomains[count+1] 
                else:
                    t_min, t_max =time_subdomains[0], time_subdomains[count+1] 
                
                # replace 0 by count if moving IC
                print("change the time domain to: ",'t_min: {0:.3f}, t_max: {1:.3f}'.format(t_min, t_max))
                flag=1

        return list_loss, results  
        ###############################################