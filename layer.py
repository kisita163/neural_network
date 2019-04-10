from perceptron import Perceptron

class Layer :
    
    def __init__(self,num_inputs,num_neurons=1,label='hidden'):
        
        self.neurons     = []
        self.label       = label
        self.init_neurons(num_neurons,num_inputs)
        
    
    
    def init_neurons(self,num_neurons,num_inputs):
        
        #print('creating neurons...')
        
        for k in range(1,num_neurons + 1) : 
            
            #print('creating neuron ' + str(k))
            n = Perceptron(num_inputs,label=self.label)
            self.neurons.append(n)
    
    
    def get_neurons(self):
        return self.neurons 
    
        
    
    def inout(self,input_array):
        
        if self.label == 'input':
        
            output_array  = []
            
            for n,i in zip(self.neurons, input_array) :
                o = n.inout([i])
                output_array.append(o)
                
        else : 
            
            output_array  = []
                        
            for n in self.neurons :
                o = n.inout(input_array)
                output_array.append(o)
            
        
        return output_array
    
    
