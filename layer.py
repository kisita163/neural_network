from perceptron import Perceptron

class Layer :
    
    def __init__(self,num_inputs,num_neurons=1):
        
        self.neurons     = []
        self.init_neurons(num_neurons,num_inputs)
        
    
    
    def init_neurons(self,num_neurons,num_inputs):
        
        print('creating neurons...')
        
        for k in range(1,num_neurons + 1) : 
            
            print('creating neuron ' + str(k))
            n = Perceptron(num_inputs)
            self.neurons.append(n)
    
    
    def get_neurones(self):
        return self.neurons 
    
        
    
    def inout(self,input_array):
        
        output_array  = []
        
        for n in self.neurons :
            o = n.inout(input_array)
            output_array.append(o)
        
        return output_array