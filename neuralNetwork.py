from perceptron import Perceptron  
from layer import Layer
  
        
train_in = [[1,1,1],
            [1,0,1],
            [1,1,0],
            [0,1,1],
            [1,0,0],
            [0,1,0],
            [0,0,1],
            [0,0,0],
            ]

train_out = [[1],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],
             [0],]


#Layers

l1 = Layer(3,num_neurons=2) # input  layer

l2 = Layer(2,num_neurons=2) # hidden layer

l3 = Layer(2)               # output layer


# training

#connections

o1 = l1.inout(train_in[0])

o2 = l2.inout(o1)

o3 = l3.inout(o2)


