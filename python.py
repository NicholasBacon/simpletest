import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt



workset = set()
process = 0

f1 = open("files/mixed", "r")

for x in f1:
    workset.add(int(x.split("-")[2]))
    process= max(int(x.split("-")[1]),process)
for work in workset:
    f1 = open("files/mixed", "r")
    xs = [11] * process
    for x in f1:
        if x.startswith("p") and int(x.split("-")[2]) == work :
            xs[int(x.split("-")[1])-1]=  min(float(x.split(",")[3].replace("\n","")),xs[int(x.split("-")[1])-1])

    x = [x/xs[0] for x in xs][1:]
    y = [x+1 for x in range(len(xs))][1:]

    plt.plot(y, x,**{'color': 'green', 'marker': 'o'},label='4 task per Node -handpack')


    f1 = open("files/mixed", "r")
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
    f1 = open("files/mixed", "r")
    xs = [0] * process
    for x in f1:
        if x.startswith("p") and int(x.split("-")[2]) == work :
            xs[int(x.split("-")[1])-1]=  max(float(x.split(",")[3].replace("\n","")),xs[int(x.split("-")[1])-1])

    x = [x/xs[0] for x in xs][1:]
    y = [x+1 for x in range(len(xs))][1:]

    plt.plot(y, x,**{'color': 'green', 'marker': 'o'},label='4 task per Node -handpack')


    f1 = open("files/mixed", "r")
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






# f4c = open("speed up/4c", "r")
# f4d = open("speed up/4d", "r")
# f4hp = open("speed up/4hp", "r")
# f9c = open("speed up/9c", "r")
# f9d = open("speed up/9d", "r")
# f9hp = open("speed up/9hp", "r")
# f16c = open("speed up/16c", "r")
# f16d = open("speed up/16d", "r")
# f16hp = open("speed up/16hp ", "r")
#
# f1speed={}
# f4cspeed={}
# f4dspeed={}
# f9cspeed={}
# f9dspeed={}
# f16cspeed={}
# f16dspeed={}
# f4hpspeed={}
# f9hpspeed={}
# f16hpspeed={}
#
# for x in f1:
#     f1speed[x.split(",")[0]]=   x.split(",")[3].replace("\n","")
# for x in f4c:
#     f4cspeed[x.split(",")[0]]=   x.split(",")[3].replace("\n","")
# for x in f4d:
#     f4dspeed[x.split(",")[0]]=   x.split(",")[3].replace("\n","")
# for x in f9c:
#     f9cspeed[x.split(",")[0]]=   x.split(",")[3].replace("\n","")
# for x in f9d:
#     f9dspeed[x.split(",")[0]]=   x.split(",")[3].replace("\n","")
# for x in f16c:
#     f16cspeed[x.split(",")[0]]=   x.split(",")[3].replace("\n","")
# for x in f16d:
#     f16dspeed[x.split(",")[0]]=   x.split(",")[3].replace("\n","")
#
#
# for x in f4hp:
#     f4hpspeed[x.split(",")[0]]=   x.split(",")[3].replace("\n","")
# for x in f9hp:
#     f9hpspeed[x.split(",")[0]]=   x.split(",")[3].replace("\n","")
# for x in f16hp:
#     f16hpspeed[x.split(",")[0]]=   x.split(",")[3].replace("\n","")
#
#
#
#
# xpoints =[4,9,16]
#
# c =[
# float(f4cspeed["c-25600"])    /float(f1speed["c-102400"]),
# float(f9cspeed["c-12800"])    /float(f1speed["c-102400"]),
# float(f16cspeed["c-6400"])    /float(f1speed["c-102400"])]
#
# d =[float(f1speed["c-102400"]),
# float(f4dspeed["c-25600"])*4,
# float(f9dspeed["c-12800"])*9,
# float(f16dspeed["c-6400"])*16]
#
# hp = [
# float(f4hpspeed["c-25600"])      /float(f1speed["c-102400"]),
# float(f9hpspeed["c-12800"])      /float(f1speed["c-102400"]),
# float(f16hpspeed["c-6400"])      /float(f1speed["c-102400"])]
#
# print(hp)
# plt.plot(xpoints, c)
# # plt.plot(xpoints, d)
# plt.plot(xpoints, hp)
# plt.show()


#
#
# for x in range(5):
#
#     d4 = []
#     d9 = []
#     d16 = []
#     fig = plt.figure(figsize=(10, 7))
#
#
#     # if x==1:
#     #
#     #     for x in f16dspeed.keys():
#     #         plt.title("Speed up amount% , (1 process) / ( mixed continuous and data type data movement) ")
#     #         d4.append(1 / (float(f4dspeed[x]) / float(f1speed[x])))
#     #         d9.append(1 / (float(f9dspeed[x]) / float(f1speed[x])))
#     #         d16.append(1 / (float(f16dspeed[x]) / float(f1speed[x])))
#     if  x==2:
#         plt.title("Speed up amount% , (1 process) / (ideal communication data movement) ")
#         for x in f16dspeed.keys():
#             d4.append(1 / (float(f4cspeed[x]) / float(f1speed[x])))
#             d9.append(1 / (float(f9cspeed[x]) / float(f1speed[x])))
#             d16.append(1 / (float(f16cspeed[x]) / float(f1speed[x])))
#     # elif x == 3:
#     #     for x in f16dspeed.keys():
#     #         plt.title(
#     #             "Speed up amount% , (all continuous data movement) / ( mixed continuous and data type data movement) ")
#     #         d4.append(1/(float(f4dspeed[x]) / float(f4cspeed[x])))
#     #         d9.append(1/(float(f9dspeed[x])/float(f9cspeed[x])))
#     #         d16.append(1/(float(f16dspeed[x]) / float(f16cspeed[x])))
#     elif x == 4:
#         for x in f16dspeed.keys():
#             plt.title(
#                 "Speed up amount% , (ideal communication data movement) / ( hand packed data movement) ")
#             d4.append(1/(float(f4hpspeed[x]) / float(f4cspeed[x])))
#             d9.append(1/(float(f9hpspeed[x])/float(f9cspeed[x])))
#             d16.append(1/(float(f16hpspeed[x]) / float(f16cspeed[x])))
#     elif x == 0:
#         for x in f16dspeed.keys():
#             plt.title(
#                 "Speed up amount% , (1 process) / ( hand packed data movement) ")
#             d4.append(1/(float(f4hpspeed[x]) / float(f1speed[x])))
#             d9.append(1/(float(f9hpspeed[x])/float(f1speed[x])))
#             d16.append(1/(float(f16hpspeed[x]) / float(f1speed[x])))
#
#     data = [[],d4,d9,d16]
#
#
#
#     frame1 = plt.gca()
#     frame1.axes.xaxis.set_ticklabels([])
#     frame1.axes.yaxis.set_ticklabels([])
#
#     ax = fig.add_subplot(111)
#     ax.set_xticklabels(['1', '4', '9', '16'])
#
#     ax.set_ylim([0, 1.2])
#
#     plt.xlabel("Processes")
#     plt.ylabel("speed up ")
#
#
#
#     plt.boxplot(data)
#
#     # show plot
#     plt.show()
