'''
Written by Pramesh Kumar (kumar372@umn.edu)
Adaptive park-and-ride choice location
'''


import itertools as it
import numpy as np

class Nodes:
    def __init__(self, _tmpIn):
        self.nodeId = _tmpIn[0]
        self.lat = float(_tmpIn[1])
        self.lon = float(_tmpIn[2])
        self.inNodes = []
        self.outNodes = []


class Links:
    def __init__(self, _tmpIn):
        self.fromNode = _tmpIn[0]
        self.toNode = _tmpIn[1]
        self.type =  _tmpIn[2]
        self.state = _tmpIn[3]
        self.depTime = int(_tmpIn[4])
        self.prob = float(_tmpIn[5])
        self.travelTime = float(_tmpIn[6])
        self.length = float(_tmpIn[7])
        self.tripId = _tmpIn[8]


class States:
    def __init__(self, _tmpIn):
        self.nodeId = _tmpIn[0]
        self.time = int(_tmpIn[1])
        self.actions = []
        self.costs = []
        self.prob = []




def readData():
    inFile = open("onlineData.dat")
    tmpIn = inFile.readline().strip().split("\t")
    for x in inFile:
        tmpIn = x.strip().split("\t")
        if tmpIn[2] not in nodeSet:
            nodeSet[tmpIn[2]] = Nodes([tmpIn[2], tmpIn[4], tmpIn[5]])
        if tmpIn[3] not in nodeSet:
            nodeSet[tmpIn[3]] = Nodes([tmpIn[3], tmpIn[6], tmpIn[7]])
        if tmpIn[3] not in nodeSet[tmpIn[2]].outNodes:
            nodeSet[tmpIn[2]].outNodes.append(tmpIn[3])
        if tmpIn[2] not in nodeSet[tmpIn[3]].inNodes:
            nodeSet[tmpIn[3]].inNodes.append(tmpIn[2])
        # A link Id is composed of from node, top node, time of departure, congested/uncongested
        linkId = (tmpIn[2], tmpIn[3], int(tmpIn[9]), tmpIn[10], "None")
        if linkId not in linkSet:
            tmpVal = [tmpIn[2], tmpIn[3], tmpIn[1], tmpIn[10], tmpIn[9], tmpIn[11], tmpIn[12], tmpIn[8], "None"]
            linkSet[linkId] = Links(tmpVal)
    inFile.close()

def readDelay():
    inFile = open("busDelay.dat")
    tmpIn = inFile.readline().strip().split("\t")
    for x in inFile:
        tmpIn = x.strip().split("\t")
        if (tmpIn[3], tmpIn[4]) not in nodeSet:
            nodeSet[(tmpIn[3], tmpIn[4])] = Nodes([(tmpIn[3], tmpIn[4]), 0, 0])
            nodeSet[(tmpIn[3], tmpIn[4])].inNodes.append(tmpIn[6])
            nodeSet[tmpIn[6]].outNodes.append((tmpIn[3], tmpIn[4]))
            nodeSet['291'].inNodes.append((tmpIn[3], tmpIn[4]))
            nodeSet[(tmpIn[3], tmpIn[4])].outNodes.append('291')

        for t in range(int(tmpIn[5]) - 600, int(tmpIn[5])+30, 30):
            # This link Id is composed of from link, to link, dep Time, state, and tripId which is the value of delay
            linkId = (tmpIn[6], (tmpIn[3], tmpIn[4]), t, tmpIn[1], tmpIn[8])
            if linkId not in linkSet:
                linkSet[linkId] = Links([tmpIn[6], (tmpIn[3], tmpIn[4]), "Waiting", tmpIn[1], t, tmpIn[2], (int(tmpIn[5]) - t), 0.1, tmpIn[8]])
        if ((tmpIn[3], tmpIn[4]), '291', int(tmpIn[5]), 'u', tmpIn[8]) not in linkSet:
            linkSet[((tmpIn[3], tmpIn[4]), '291', int(tmpIn[5]), 'u', tmpIn[8])] = Links([(tmpIn[3], tmpIn[4]), '291', "Transit", 'u', tmpIn[5], 1.0, tmpIn[7], 0.0, tmpIn[8]])
    inFile.close()


def readSchedule():
    inFile = open("scheduleTransit.dat")
    tmpIn = inFile.readline().strip().split("\t")
    for x in inFile:
        tmpIn = x.strip().split("\t")
        if (tmpIn[4], tmpIn[3]) not in nodeSet:
            nodeSet[(tmpIn[4], tmpIn[3])] = Nodes([(tmpIn[4], tmpIn[3]), 0, 0])
            nodeSet[(tmpIn[4], tmpIn[3])].inNodes.append(tmpIn[1])
            nodeSet[tmpIn[1]].outNodes.append((tmpIn[4], tmpIn[3]))
            nodeSet[(tmpIn[4], tmpIn[3])].outNodes.append('291')
            nodeSet['291'].inNodes.append((tmpIn[4], tmpIn[3]))
        # This linkId is composed of from node, to node, departure time, state
        for t in range(int(tmpIn[5]) - 600 , int(tmpIn[5])+30, 30):
            linkId = (tmpIn[1], (tmpIn[4], tmpIn[3]), t, '0', tmpIn[7])
            if linkId not in linkSet:
                linkSet[linkId] = Links([tmpIn[1], (tmpIn[4], tmpIn[3]), "Waiting",  '0', t, 1.0, (int(tmpIn[5]) - t), 0.1, tmpIn[7]])
        if ((tmpIn[4], tmpIn[3]), '291', int(tmpIn[5]), 'u', tmpIn[7]) not in linkSet:
            linkSet[((tmpIn[4], tmpIn[3]), '291', int(tmpIn[5]), 'u', tmpIn[7])] = Links([(tmpIn[4], tmpIn[3]), '291', "Transit", '0', tmpIn[5], 1.0, tmpIn[6], 0.0, tmpIn[7]])

    inFile.close()
    print(len(nodeSet), "nodes")
    print(len(linkSet), "links")




def createStates():
    for n in nodeSet:
        for t in range(36000, 21630, -30):
            possTrans = [a for a in linkSet if linkSet[a].fromNode == n and linkSet[a].depTime == t]
            if len(possTrans) != 0:
                outNodeStates = {(linkSet[a].toNode, linkSet[a].tripId): [] for a in possTrans}
                for l in outNodeStates:
                    tempList = [(linkSet[a].toNode, linkSet[a].depTime, linkSet[a].state, linkSet[a].travelTime, linkSet[a].prob, linkSet[a].tripId) for a in possTrans if a[1] == l[0] and a[4] == l[1]]
                    for k in tempList:
                        outNodeStates[l].append(k)
                z = outNodeStates.values()
                thetaList = list(it.product(*z))
                sumProb = 0
                for theta in thetaList:
                    p = []
                    prob = 1
                    for q in theta:
                        p.append((q[0], q[2], q[3], q[5]))
                        prob = prob*q[4]
                    p.append(round(prob, 3))
                    sumProb = sumProb + prob
                    #print((n, t, p))
                    stateSet.append((n, t, p))
                if round(sumProb) != 1:
                    print("Your probabilities don't sum to one!")
                    #print((sumProb, n, t))






def LabelSetting():
    for t in range(50010, 21630, -30):
        lastKey = sorted(statesName.keys())[-1]
        stateSet.append(('291', t, [('291', 'u', 0.0), 1]))
        statesName[lastKey + 1] = ('291', t, [('291', 'u', 0.0), 1])
        labels[('291', t)] = 0
        policy[lastKey+1] = 'NA'
        for n in nodeSet:
            if n != '291':
                labels[(n, t)] = float("inf")
                for s in [a for a in stateSet if a[0] == n and a[1] == t]:
                    findKey = [k for k, v in statesName.items() if v == s][0]
                    policy[findKey] = 'NA'
    SEL = list(set([a[0] for a in stateSet if a[0] in nodeSet['291'].inNodes]))
    print("No problems until here! :)")

    while SEL:
        i = SEL.pop(0)
        for t in range(36000, 21630, -30):
            possibleStates = [a for a in stateSet if a[0] == i and a[1] == t]
            tempJ = 0
            for st in range(len(possibleStates)):
                prob = possibleStates[st][2][-1]
                index = possibleStates[st][2].index(prob)
                tempJ = tempJ + prob * min([a[2] + labels[a[0], t + int(a[2])] for a in possibleStates[st][2][:index]])
                if tempJ < labels[(i, t)]:
                    labels[(i, t)] = tempJ
                    for st in range(len(possibleStates)):
                        prob = possibleStates[st][2][-1]
                        index = possibleStates[st][2].index(prob)
                        costs = [a[2] + labels[a[0], t + int(a[2])] for a in possibleStates[st][2][:index]]
                        policy[[k for k, v in statesName.items() if v == possibleStates[st]][0]] = \
                        possibleStates[st][2][costs.index(min(costs))][0]
                        SEL = SEL + list(set([a[0] for a in stateSet if a[0] in nodeSet[i].inNodes]))
        print(i)



def VI():
    for t in range(50010, 21630, -30):
        lastKey = sorted(statesName.keys())[-1]
        stateSet.append(('291', t, [('291', 'u', 0.0, 'None'), 1]))
        statesName[lastKey + 1] = ('291', t, [('291', 'u', 0.0, 'None'), 1])
        Jhat[('291', t)] = 0
        oldJ[lastKey + 1] = 0

    for s in statesName:
        oldJ[s] = 0

    print("Starting drama done!")

    for k in range(23):
        for t in range(36000, 21630, -30):
            for i in nodeSet:
                if i != '291':
                    possibleStates = [a for a in stateSet if a[0] == i and a[1] == t]
                    tempJhat = 0
                    for theta in possibleStates:
                        prob = theta[2][-1]
                        index = theta[2].index(prob)
                        costs = []
                        for links in theta[2][:index]:
                            transitionStates = [a for a in stateSet if a[0] == links[0] and a[1] == int(t+links[2])]
                            temp = 0
                            for each in transitionStates:
                                findKey = [k for k, v in statesName.items() if v == each][0]
                                temp += oldJ[findKey]*each[2][-1]
                            costs.append(temp + links[-2])
                        #optInd = costs.index(min(costs))

                        findKeyAgain = [k for k, v in statesName.items() if v == theta][0]
                        newJ[findKeyAgain] = round(min(costs))
                        #print("New J", newJ[findKeyAgain])
                        tempJhat += newJ[findKeyAgain]*theta[2][-1]
                    Jhat[(i, t)] = round(tempJhat)
                    #print("Jhat", Jhat[(i, t)])
            if t  == 26000:
                print("halfway through")


        count = 0
        for iter in oldJ:
            if iter in newJ:
                if oldJ[iter] != newJ[iter]:
                    count = count +1
                    oldJ[iter] = newJ[iter]
        print(count)

        if count == 0:
            continue


    for t in range(36000, 21630, -30):
        for i in nodeSet:
            if i != '291':
                possibleStates = [a for a in stateSet if a[0] == i and a[1] == t]
                for theta in possibleStates:
                    prob = theta[2][-1]
                    index = theta[2].index(prob)
                    costs = []
                    for links in theta[2][:index]:
                        transitionStates = [a for a in stateSet if a[0] == links[0] and a[1] == int(t + links[2])]
                        temp = 0
                        for each in transitionStates:
                            findKey = [k for k, v in statesName.items() if v == each][0]
                            temp += oldJ[findKey] * each[2][-1]
                        costs.append(temp + links[-2])
                    findKeyAgain = [k for k, v in statesName.items() if v == theta][0]
                    optInd = costs.index(min(costs))
                    optPolicy[findKeyAgain] = theta[2][optInd][0]














nodeSet = {}
linkSet = {}
readData()
#readDelay()
#readSchedule()
stateSet = []

labels ={}
policy = {}
statesName = {}
createStates()
statesName = {i:stateSet[i] for i in range(len(stateSet))}
#LabelSetting()

oldJ = {}
newJ = {}

Jhat = {}
optPolicy = {}
VI()





# Comparing the online shortest path with the expected a priori shortest path
import pandas as pd
df = pd.DataFrame([(k[1], labels[k]) for k in labels if k[0] == '269'])
df.to_csv("recourse", sep='\t', encoding='utf-8')

df = pd.read_csv("recourse.csv")

import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
import matplotlib.patches as mpatches # for legends


# Defining line types
from collections import OrderedDict
from matplotlib.transforms import blended_transform_factory


linestyles = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])







df  = df[df['Savings']>0]

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size = 25)

# Time components
X= df['time']
Y1 = df['Recourse']
Y2 = df['SP ']


plt.rcParams['figure.figsize'] = 20, 12
plt.plot(X, Y1,'--',color="black", label = "Adaptive routing cost")
plt.plot(X, Y2,'-',color="black", label = "A priori shortest path cost")


plt.xlabel('Time (sec)', fontsize = 25)
plt.ylabel('Travel time (sec)', fontsize = 25)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
         ncol=3, fancybox=True, shadow=True)
plt.savefig('comp.png', dpi=100)
plt.show()
plt.xticks(rotation=70)



# Creating a heat map of expected cost
df = pd.DataFrame([(k[0], k[1], Jhat[k]) for k in Jhat if k[1] <= 32910 and len(k[0]) != 2])
#df = pd.DataFrame([(k[0], k[1], labels[k]) for k in labels if k[1] <= 32910 and len(k[0]) != 2])
df.columns = ["Node", "Time", "TravelTime"]
df['Time'] = pd.to_datetime(df.Time, unit='s').dt.strftime('%H:%M:%S').astype(str).values.tolist()

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import seaborn as sns; sns.set()
plt.rc('font', family='serif', size = 25)
plt.rcParams['figure.figsize'] = 15, 5

Z = pd.pivot_table(df, values='TravelTime',index='Time',columns='Node')
ax = sns.heatmap(Z, cmap='RdYlGn_r', cbar_kws={'label': 'Travel time (sec)'}, vmin=0, vmax=1500)
ax.invert_yaxis()
ax.set(xlabel='Links', ylabel='Clock')
plt.savefig('heatExp.png', dpi=100)
plt.show()




# Sort of doing park-andride nodes assessment
import pandas as pd
pnrPlocies = [(statesName[k], optPolicy[k]) for k in optPolicy if optPolicy[k] != 'NA' and statesName[k][0]  in ['2751', '2701', '2721', '2771', '2791'] and statesName[k][1] <= 32910]

uncongested = [k for k in pnrPlocies if k[0][2][0][1] == 'u']
polU =[]
for a in uncongested:
    if len(a[1]) == 2:
        polU.append((a[0][0], a[0][1], 1))
    else:
        polU.append((a[0][0], a[0][1], 0))






df = pd.DataFrame(polC, columns=['Node', 'Time', 'Mode'])
df['Time'] = pd.to_datetime(df.Time, unit='s').dt.strftime('%H:%M:%S').astype(str).values.tolist()
Z = pd.pivot_table(df, values='Mode', index='Node', columns='Time')
Z.fillna(0, inplace=True)
import matplotlib.pyplot as plt
import pandas
import seaborn.apionly as sns
plt.rc('font', family='serif', size = 20)
plt.rcParams['figure.figsize'] = 18, 15

from matplotlib.colors import LinearSegmentedColormap
myColors = ((1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 0.0))
cmap = sns.cubehelix_palette(start=2.8, rot=.1, light=0.9, n_colors=2)
ax = sns.heatmap(Z, cmap=cmap)
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([0, 1])
#colorbar.set_yticklabels(['Take PR', 'Take Auto'])
ax.set_ylabel('PNR Node')
ax.set_xlabel('Clock')
#plt.setp(labels, rotation=0)
plt.savefig('u.png', dpi=100)
plt.show()



import pandas as pd
uncongested = [k for k in pnrPlocies if k[0][2][0][1] == 'c']
polC =[]
for a in uncongested:
    if len(a[1]) == 2:
        polC.append((a[0][0], a[0][1], 1))
    else:
        polC.append((a[0][0], a[0][1], 0))











