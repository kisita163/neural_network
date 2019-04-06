import numpy as np
import matplotlib.pyplot as plt

import array

class Perceptron :
    

    def __init__(self,input_lenght,bias=-0.1):#-44.36169230842653):
        self.outputs      = []
        self.bias         = bias
        self.input_length = input_lenght
        self.init_weights(input_lenght)
        
    
    def init_weights(self,length):
        self.weights = np.random.rand(1,length)

        
    def get_weights(self):
        return self.weights[-1]
    
    def get_weights_history(self):
        return self.weights
    
    
    def set_weights(self,new_weights):
        self.weights = np.append(self.weights,[new_weights],axis=0)

        
    def sigmoid(self,x):
        return 1 / ( 1 + np.exp(-x))  
         
    
    def sigmoid_derivative(self,x):
        return x*(1 - x)
    
    
    def get_outputs(self):
        
        return self.outputs
    
    
    def inout(self,input_array):
        
        if len(input_array) != self.input_length : 
            raise Exception('The input array length is greater or lower than the expected length ('+ str(self.input_length) + ')')

        dot_product = np.dot(input_array,self.get_weights()) + self.bias
        
        return self.sigmoid(dot_product)


    def training(self,input_arrays,expected_array,learning_rate, epoch):
        
        
        for i in range(epoch):
            print('Starting epoch (' + str(i + 1) + ')')
            print('\n')
            print("Old weights : " + str(self.get_weights()))
            o = []
            expected_index = 0  
            errors = []
            
            for array in input_arrays : 
                print('input are : '+ str(array))
                out  = self.inout(array) 
                o.append(out) 
                print('Calculated out is : ' + str(out))
                print('Expected out is   : ' + str(expected_array[expected_index][0]))
                
                weights = []
        
                
                for weight,in_array in zip(self.get_weights(),array) : 
                    
                    error = out - expected_array[expected_index][0]    
                    
                    w = weight + learning_rate*(error)*in_array
                    
                    print('w = '+str(weight) + ' + ' + str(learning_rate)+'*('+str(out)+' - ' + str(expected_array[expected_index][0])+')*'+str(input))
                    self.bias = self.bias + learning_rate*(error)
                    weights.append(w)
                
                  
                self.set_weights(weights)
                print("New weights : " + str(self.get_weights()))
                expected_index = expected_index + 1
                print('\n')
                
                errors.append(error)
                
            self.outputs.append(o)
            print('-----------------------------------------------')
            print('\n\n\n')
            if np.amax(errors) < 0.02 : 
                break
        print('bias is ' + str(self.bias))
       
        return i
            
    
    def plot_output(self,k,train_index):
        
        A = np.array(self.get_outputs())
        t = np.arange(0,k+1,1)
       
        
        plt.close('all')
        plt.plot(t,A[:,train_index],'r--')
        plt.show()



