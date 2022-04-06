import numpy as np
from math import sin, cos, pi, exp
from matplotlib import pyplot

a = np.array([[1,2,3,4],[5,6,7,8],[9,8,7,6],[5,4,3,2]])
b = np.array([[1,2,3,4],[5,6,7,8],[9,8,7,6]])
c = np.array([[1,2,3],[5,6,7],[9,8,7],[5,4,3]])
d = np.array([[6,2,4],[3,6,8],[1,8,6]])

a_c = np.concatenate((a,c),axis = 1)
b_d = np.concatenate((b,d),axis =1)
e = np.concatenate((a_c, b_d),axis = 0)
# inverse_e = np.linalg.inv(e)
det_e = det = np.linalg.det(e)


print("answer of 1-(a) is", e)
# print("answer of 1-(b) is", inverse_e)
print("answer of 1-(c) is", det_e)
x = [0.001*i for i in range(-50,50)]
y1 = (sin(2*pi*300*t-pi/3)**2 for t in x)
y2 = (exp(cos(2*pi*100*t-pi/3)) for t in x) 

pyplot.plot(x,y1)
pyplot.show()
# pyplot.plot(x,y2)
# pyplot.show()