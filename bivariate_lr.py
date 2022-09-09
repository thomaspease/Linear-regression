import numpy as np

x_train = np.array([1,2,3,4,5])
y_train= np.array([0.48,1,1.52,2.03,2.46])

def compute_cost(x, y, w, b):
  m = x.shape[0]
  cost=0

  for i in range(m):
    f_wb = (x[i] * w) + b
    cost += 1/(2*m) * ((f_wb - y[i])**2)
  
  print(cost)

def gradient_descent(a, w, b, x, y, iterations):
  def gradient(w,b,x,y,option):
    m=x.shape[0]
    cost_gradient=0

    for i in range(m):
      if option == 0:
        cost_gradient += (1/m) * (((x[i] * w) +b) - y[i])
      if option == 1:
        cost_gradient += (1/m) * ((((x[i] * w) +b) - y[i]) * x[i])
    return cost_gradient

  for i in range(iterations):
    new_w = w - a * gradient(w,b,x,y,1)
    new_b = b - a * gradient(w,b,x,y,0)
    w = new_w
    b = new_b

  return(w,b)
  
# Testing the programme to show the cost decreasing after gradient descent is run
w = 1
b = 1

compute_cost(x_train, y_train, w, b)
w,b = gradient_descent(0.01, w, b, x_train, y_train, 10000)
compute_cost(x_train, y_train, w, b)

