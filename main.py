from neural_network import Neural

        
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


my_network = Neural(num_layers=3,num_neurons=[3,5,3])

    
my_network.training(train_in, train_out, epoch=5000)


print('\n')
print('Verification')
for v in verification_in :
    print(my_network.inout(v))
    
    
    
for l in my_network.get_layers() : 
    print('\n')
    print('label is '  + l.get_label())
    print('Number of neurons is : ' + str(len(l.get_neurons())))
    for n in l.get_neurons() : 
        print('weigths = ' + str(n.get_weights()))
    print('\n')
    print('\n')
    