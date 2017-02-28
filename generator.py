import random

x_center,y_center = 10,10

data = list()
num = 2000
inside,outside = 0,0
while (inside + outside < num):
  point = list()
  point.append(random.uniform(0,20))
  point.append(random.uniform(0,20))
  if (((point[0] - x_center)**2 + (point[1] - y_center)**2 < 36) and inside < num/2): 
    point.append(1)
    inside +=1
    data.append(point)
  elif (outside < num/2) : 
    point.append(0)
    outside +=1
    data.append(point)
for row in data:
  print (row)
