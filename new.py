import numpy as np

x_train = np.array([1,2,3,4,5])
y_train= np.array([0.48,1,1.52,2.03,2.46])

def compute_cost(x, y, w, b):
  m = x.shape[0]
  cost=0

  for i in range(m):
    f_wb = (x[i] * w) + b
    cost += 1/(2*m) * ((f_wb - y[i])**2)

  return cost

# print_cost = compute_cost(x_train, y_train, 1, 1)



def gradient_descent(a, w, b, x, y, iterations):
  def w_gradient(w,b,x,y):
    m=x.shape[0]
    cost_gradient=0

    for i in range(m):
      cost_gradient += (1/m) * ((((x[i] * w) +b) - y[i]) * x[i])

    return cost_gradient

  def b_gradient(w,b,x,y):
    m=x.shape[0]
    cost_gradient=0

    for i in range(m):
      cost_gradient += (1/m) * (((x[i] * w) +b) - y[i])
    return cost_gradient

  for i in range(iterations):
    new_w = w - a * w_gradient(w,b,x,y)
    new_b = b - a * b_gradient(w,b,x,y)
    w = new_w
    b = new_b
    print(w)
    print(b)

gradient_descent(0.01, 0.8, 1, x_train, y_train, 10000)
