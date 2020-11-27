# -*- coding: utf-8 -*-
"""grad_descnt_harsh.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/120uB7gbnLqgA7OBsyEVzIeqPztIyJf20
"""

#training set 
x_train = [1.0, 2.0, 3.0]
y_train = [3.0, 8.0, 15.0]

#starting values
w1=1.0;    #correct value is 2.0
w2=2.0;    #correct value is 1.0 

#forward pass
def forward(x):
    return (x*x*w2 + x*w1)

# calculating loss
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

# compute gradient wrt w1
def gradient1(x, y):  
    return 2 * x * (x*w1 + x*x*w2 - y)

# compute gradient wrt w2
def gradient2(x, y):  
    return 2* x* x* (x*w1 + x*x*w2 - y)    

# Before training 
print("Prediction (before training) for x=",4, "is:", forward(4))
# correct value is 24.0

# Training 
for epoch in range(500):
    for x_val, y_val in zip(x_train, y_train):
        
        grad1 = gradient1(x_val, y_val)
        w1 = w1 - 0.01 * grad1

        grad2 = gradient2(x_val, y_val)
        w2 = w2 - 0.01 * grad2
        print("\tgrad: ", x_val, y_val, round(grad1, 2), round(grad2, 2))
        l = loss(x_val, y_val)
    print("epoch:", epoch, "w1=", round(w1, 2),  "w2=", round(w2, 2), "loss=", round(l, 2))

# After training
print("Prediction (after training) for x=",4, "is:", forward(4))