from perceptron import Perceptron  
  
        
train_in = [
    [1, 1],
    [0.6,0.6],
    [1, 0],
    [0, 1],
    [0, 1],
    [0, 0],
    [0.9, 0.9],
    [0.3, 0.2],
    [0.9, 0.3],
    [0.6,0.9]]

test_in = [
    [1, 1],#0
    [1, 0],#0
    [0, 1],#0
    [0, 0],#0
    ]#1
 
#output
train_out = [
[1],
[1],
[0],
[0],
[0],
[0],
[1],
[0],
[0],
[1]]



p = Perceptron(2)

k = p.training(train_in,train_out,-0.8,100)

print(p.get_weights_history())

#p.plot_output(k,0)

for v in test_in :
    k = p.inout(v)
    
    if k > 0.5 :
        print(1)
    else :
        print(0)
