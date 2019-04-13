from layer import Layer



#Layers

class Neural : 
    
    def __init__(self,num_layers=2,num_neurons = [3,1],debug=True):
        
        '''
            num_layers  : Number of layers
            num_neurons : Array containg number of neurons in each layer
        '''
        
        self.num_layers  = num_layers
        self.num_neurons = num_neurons   
        self.debug       = debug
        
        self.__init_layers()
        
        
    def traces(self,s):
        if self.debug == True : 
            print(s)
        
    
    def __init_layers(self):
        
        self.layers = []
        n_1= None
        
        for i,n in zip(range(1,self.num_layers + 1),self.num_neurons) : 
            
            #self.traces('Number of neurons in above layer = ' + str(n_1))
            
            if i == 1 : #Input layer
                l = Layer(1,label='input',num_neurons=n)
            elif i == self.num_layers : #Output layer
                l = Layer(n_1,label='output',num_neurons=n)
            else : #Hidden layer    
                l = Layer(n_1,label='hidden',num_neurons=n)
                
            self.layers.append(l)
            
            n_1 = n
            
    
    def get_layers(self):
        
        return self.layers
    
    
    def inout(self,vector_in):
        
        '''
            Returns the output of the neural network
            
            vector_in  : Input vector
            vector_out : Output vector
        '''
        
        vector_out = None
        
        for l in self.layers : 
            #self.traces(str(vector_out) + ' ' + l.get_label())
            if l.get_label() == 'input' : 
                vector_out = l.inout(vector_in)
            else : 
                vector_out = l.inout(vector_out)    
        
        return vector_out
    
    
    def training(self,train_in,train_out,epoch=100):
        
        '''
            train_in  : Input array
            train_out : Desired output array
            epoch     : Number of loop for training the neural network
        '''

        for epoch in range(0,epoch):
            
            index  = 1
            
            
            for v,d in zip(train_in,train_out) : 
                
                self.traces('Processing index ' + str(index) + '('+str(epoch)+')')
                
                self.inout(v)

                #back propagation
                
                above_errors = None
                
                for l in reversed(self.get_layers()) :
                    # compute error for each neurons of this layer
                    errors = []
                    
                    i = 0
                    
                    for n in l.get_neurons() : 
                        if l.get_label() == 'output': 
                            
                            e = n.update_weight(desired=d[i])
                            
                        else : 
                        
                            e = n.update_weight(above_errors=above_errors)
                    
                        errors.append(e)
                        
                        i = i + 1
                            
                    above_errors  = errors  
                    
                index = index  + 1    
            