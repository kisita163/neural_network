
from layer import Layer
import numpy as np
import matplotlib.pyplot as plt

  
        
train_in = [[1,1,1],
            [1,0,0],
            [0,1,0],
            [1,1,0],
            [0,1,1],
            [0,0,1],
            [1,0,1],
            [0,0,0],]

verification_in = [[1,0.8,1],
            [0.7,0,0],
            [0,10,0],
            [0.6,0.6,0],
            [0,1,1],
            [0,0,1],
            [1,0.6,0.8],
            [0,0,0],]


train_out = [[1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],]


#Layers

l1 = Layer(1,label='input',num_neurons=3) # input  layer

l2 = Layer(3,label='output')# output layer


error_out = []

epoch = 2000

# training

for epoch in range(0,epoch):
    
    for v,d in zip(train_in,train_out) : 
        
        o1 = l1.inout(v)
        o2 = l2.inout(o1)
      
        #back propagation
        
        # output layer 
        print('output layer')
        print('============')
        
        for n in l2.get_neurons() : 
            e_out  = n.update_weight(desired=d[0])
            
        
        print('\n')   
        
                # output layer 
        print('Input Layer')
        print('===========')
        
        
        for n,e in zip(l1.get_neurons(),e_out) : 
            
            e_in  = n.update_weight(above_errors=[e])
        
        print('\n')
        
    error_out.append(e_out)

print('\n')
print('\n')

#t = np.arange(0,epoch,1)


#error_out = np.array(error_out)
#print(error_out)
#plt.plot(t,error_out[:,1])

#plt.show()


print('verification')   
#connections
for v,d in zip(verification_in,train_out) : 
    
    o1 = l1.inout(v)
    o2 = l2.inout(o1)
    print(o2)


