import math

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]
 
loss = -(softmax_output[0]*math.log(softmax_output[0])+
         softmax_output[1]*math.log(softmax_output[1])+
         softmax_output[2]*math.log(softmax_output[2]))

print(loss)