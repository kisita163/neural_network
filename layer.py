from perceptron import Perceptron

class Layer :
    
    def __init__(self,num_neurones=1,num_inputs):
        
        self.neurones     = []
        self.output_array = []
        self.init_neurones(num_neurones,num_inputs)
        
    
    
    def init_neurones(self,num_neurones,num_inputs):
        
        for k in range(0,num_neurones-1,num_inputs) : 
            
            n = Perceptron(num_inputs)
            self.neurones.append(n)
    
    
    def get_neurones(self):
        return self.neurones 
        
    
    def inout(self,input_array):
        
        for n in self.neurones :
            o = n.inout(input_array)
            self.output_array.append(o)