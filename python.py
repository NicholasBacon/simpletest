import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt



workset = set()
process = 0

f1 = open("patcodefiles/mixed", "r")

for x in f1:
    workset.add(int(x.split("-")[2]))
    process= max(int(x.split("-")[1]),process)
for work in workset:
    f1 = open("patcodefiles/mixed", "r")
    xs = [11] * process
    for x in f1:
        if x.startswith("p") and int(x.split("-")[2]) == work :
            xs[int(x.split("-")[1])-1]=  min(float(x.split(",")[3].replace("\n","")),xs[int(x.split("-")[1])-1])

    x = [x/xs[0] for x in xs][1:]
    y = [x+1 for x in range(len(xs))][1:]

    plt.plot(y, x,**{'color': 'green', 'marker': 'o'},label='4 task per Node -handpack')


    f1 = open("patcodefiles/mixed", "r")
    xs = [11] * process
    for x in f1:
        if x.startswith("np") and int(x.split("-")[2]) == work:
             xs[int(x.split("-")[1])-1]=  min(float(x.split(",")[3].replace("\n","")),xs[int(x.split("-")[1])-1])

    x = [x/xs[0] for x in xs][1:]
    y = [x+1 for x in range(len(xs))][1:]

    plt.plot(y, x,**{'color': 'red', 'marker': 'o'},label='4 task per Node,no pack just send')


    plt.legend()
    plt.title(str(work)+" work- 1 send-min")
    plt.xlabel("ranks^2")
    plt.ylabel("Speed up")

    plt.show()



for work in workset:
    f1 = open("patcodefiles/mixed", "r")
    xs = [0] * process
    for x in f1:
        if x.startswith("p") and int(x.split("-")[2]) == work :
            xs[int(x.split("-")[1])-1]=  max(float(x.split(",")[3].replace("\n","")),xs[int(x.split("-")[1])-1])

    x = [x/xs[0] for x in xs][1:]
    y = [x+1 for x in range(len(xs))][1:]

    plt.plot(y, x,**{'color': 'green', 'marker': 'o'},label='4 task per Node -handpack')


    f1 = open("patcodefiles/mixed", "r")
    xs = [0] * process
    for x in f1:
        if x.startswith("np") and int(x.split("-")[2]) == work:
             xs[int(x.split("-")[1])-1]=  max(float(x.split(",")[3].replace("\n","")),xs[int(x.split("-")[1])-1])

    x = [x/xs[0] for x in xs][1:]
    y = [x+1 for x in range(len(xs))][1:]

    plt.plot(y, x,**{'color': 'red', 'marker': 'o'},label='4 task per Node,no pack just send')


    plt.legend()
    plt.title(str(work)+" work- 1 send-max")
    plt.xlabel("ranks^2")
    plt.ylabel("Speed up")

    plt.show()



