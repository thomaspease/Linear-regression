import numpy as np

# First column of x is distance from tube, and second column is sq ft. 
# Taken from recent house sales in n15
x_train= np.array([[8,836], [13,755], [22,1239], [9,1174]])
y_train= np.array([596, 500, 637, 681])

normalised_x = []
maxes = []

def normalise(ar):
  global maxes, normalised_x
  maxes = np.amax(ar, axis=0)
  normalised_x = np.array([i/maxes for i in ar])
  
normalise(x_train)

def multiple_gradient_descent(x, y, w, b, a, iterations):
  m = x.shape[0]
  def cost():
    cost = 0
    for i in range(m):
      cost += 1/2*m * (((np.dot(w,x[i]) + b) - y[i])**2)
    print(cost)

  def cost_gradient(position):
    gradient = 0
    for i in range(m):
      if position is not None:
        gradient += (((np.dot(w,x[i]) + b) - y[i]) * x[i, position])
      else:
        gradient += ((np.dot(w,x[i]) + b) - y[i])
    return a * ((1/m) * gradient)

  def update_values(w,b):
    b = b - cost_gradient(None)
    for i in range(len(w)):
      w[i] = w[i] - cost_gradient(i)
    return w,b

  # Run GD and show change in cost
  print('Cost before GD = ')
  cost()
  
  for i in range(iterations):
    w,b = update_values(w,b)
    # print(w,b)
  
  print('Cost after GD = ')
  cost()

multiple_gradient_descent(normalised_x, y_train, [0.1, 0.1], 0.1, 0.1, 10000)

