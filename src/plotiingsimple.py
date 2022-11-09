import matplotlib.pyplot as plt

X, Y = [], []
i = 0
for line in open('learning.data', 'r'):
  values = [float(s) for s in line.split()]
  X.append(i)
  Y.append(values)
  i += 1 

plt.plot(X, Y)
plt.savefig('myplot.png')