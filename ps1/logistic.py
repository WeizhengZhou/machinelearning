import numpy as np
import matplotlib.pyplot as plt
from matplotlib  import cm



def Sigmod(theta, x):
  return 1.0 / (1.0 + np.exp(-np.dot(theta, x)))


def LogisticRun(theta, x, y, alpha):
  return theta + alpha * (y - Sigmod(theta, x)) * x


def ReadX():
  x_list = []
  f = open('x.txt')
  for row in f.readlines():
    data = [c for c in row.split(' ') if c]
    x1 = float(data[0])
    x2 = float(data[1][:-1])
    x_list.append(
      np.array([x1, x2])
    )
  return x_list
  
def ReadY():
  y_list = []
  f = open('y.txt')
  for row in f.readlines():
    data = [c for c in row.split(' ') if c]
    y = float(data[0][:-1])
    y_list.append(np.array(y))
  return y_list

  
def main():
  alpha = 1
  x_list = ReadX()
  y_list = ReadY()
  
  theta = np.array([0, 0])

  x1_list = [x[0] for x in x_list]
  x2_list = [x[1] for x in x_list]
  
  
  for x, y in zip(x_list, y_list):
    theta = LogisticRun(theta, x, y, alpha)
    print(theta)
  
  plt.scatter(x1_list, x2_list, 60, y_list)
  plt.show()

  x1_list = np.linspace(min(x1_list), max(x1_list))
  x2_list = np.linspace(min(x2_list), max(x2_list))
  y_list = [Sigmod(theta, x1, x2) for (x1, x2) in zip(x1_list, x2_list)]
  # plt.plot(x1_list, x2_list)
  # plt.show()

if __name__ == '__main__':
  main()