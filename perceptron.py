import numpy as np

import array

class Perceptron :
    

    def __init__(self,input_lenght,bias=-0.0,label='output',debug=False):
        self.label        = label
        self.outputs      = []
        self.bias         = bias
        self.input_length = input_lenght
        
        self.last_in      = None
        self.last_out     = None
        self.debug        = debug 
        
        self.error_history=[] 
        
        self.init_weights(input_lenght)
     
    def traces(self,s):
        if self.debug == True : 
            self.traces(s)
            
    def init_weights(self,length):
        self.weights = np.random.rand(2,length) - 0.5
        
             
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
        
        out = self.sigmoid(dot_product)
        
        self.last_in  = input_array
        self.last_out = out
        
        return out


    def training(self,input_arrays,expected_array,learning_rate, epoch):
        
        '''
            This function is used to train the node when used individually to take decision
            
            input_arrays   = Training vectors
            expected_array = expected results array
            learning_rate  = The learning rate
            epoch          = Number of iteration
        '''
        
        for i in range(epoch):
            self.traces('Starting epoch (' + str(i + 1) + ')')
            self.traces('\n')
            self.traces("Old weights : " + str(self.get_weights()))
            o = []
            expected_index = 0  
            errors = []
            
            for array in input_arrays : 
                self.traces('input are : '+ str(array))
                out  = self.inout(array) 
                o.append(out) 
                self.traces('Calculated out is : ' + str(out))
                self.traces('Expected out is   : ' + str(expected_array[expected_index][0]))
                
                weights = []
        
                
                for weight,in_array in zip(self.get_weights(),array) : 
                    
                    error = out - expected_array[expected_index][0]    
                    
                    w = weight + learning_rate*(error)*in_array
                    
                    self.traces('w = '+str(weight) + ' + ' + str(learning_rate)+'*('+str(out)+' - ' + str(expected_array[expected_index][0])+')*'+str(input))
                    
                    #self.bias = self.bias + learning_rate*(error)
                    weights.append(w)
                
                  
                self.set_weights(weights)
                self.traces("New weights : " + str(self.get_weights()))
                expected_index = expected_index + 1
                self.traces('\n')
                
                errors.append(error)
                
            self.outputs.append(o)
            self.traces('-----------------------------------------------'  )
            self.traces('\n\n\n')
            if np.amax(errors) < 0.02 : 
                break
        self.traces('bias is ' + str(self.bias))
       
        return i
    
    def update_weight(self,learning_rate_1 = 0.8 ,learning_rate_2 = 0.8 , desired = 0 , above_errors = []):
        
        weights = []
        
        self.traces('\n')
        self.traces('initial weights')
        self.traces('\n')
        self.traces(self.get_weights())
        self.traces('\n')
        
        e   = self.get_error(desired, above_errors)
        
        for w_,w__,i in zip(self.get_weights(),self.get_weights_history()[-2], self.last_in) : 
            
            w   = w_ + learning_rate_1 * e * i  + learning_rate_2 * (w_ - w__)
            
            self.bias = self.bias + learning_rate_1*(e)  + learning_rate_2 * (w_ - w__)
            
            self.traces('w = ' + str(w_) + ' + ' + str(learning_rate_1) + '*' + str(e) + '*' + str(i) + ' + ' + str(learning_rate_2) + '*(' + str(w_) + ' - ' + str(w__) + ')')
            self.traces('\n')
            
    
            weights.append(w)
    
        self.set_weights(weights)
        
        self.traces('New weigths are : ')
        self.traces('\n\t'  +  str(self.get_weights()))
        
        self.traces('\n')
        
        self.traces('New error is : ')
        self.traces('\n\t' + str(e))
        
        return e
    
    def get_error_history(self):
        
        return self.error_history
            
    
    def get_error(self,desired=0,above_errors=[]):
        
        '''  
            desired      = desired output of the neurons. Used only when this neuron is an output neuron
            above_errors = errors  from the layer above. Here I consider the product w_jk*sigma_k
        '''
        
        #self.traces('Computing error for this ' + self.label + ' neuron')
        #self.traces('last_out = ' + str(self.last_out))
        #self.traces('desired  = ' + str(desired))
        
        if self.label == 'output':
            
            error = self.last_out*(1 - self.last_out)*(desired - self.last_out)
            
            self.traces('error = '+ str(self.last_out) + '* (1 - ' + str(self.last_out) + ')*( ' + str(desired) + ' - ' + str(self.last_out) + ')')
            self.traces('\n')
        else :
            error = self.last_out*(1 - self.last_out)*sum(above_errors)
            self.traces('error = '+ str(self.last_out) + '* (1 - ' + str(self.last_out) + ')*' + str(sum(above_errors)))
            self.traces('\n')
        
        self.error_history.append(error)  
        return error
            
            
            
            


