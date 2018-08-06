#/usr/bin/env python
#coding=utf-8
import sys,math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold





clusterindex = 0
questionindex = {}
clusterindexsim = {}
clusterinfo = {}
questionfalse = {}
count = 0
limit = 2000000
for line in open('../input/train.csv'):
    if not line:
        continue
    array = line.strip().split(',')
    if array[0] == 'label':
        continue
    if count < limit:
        if questionindex.get(array[1],-1)==-1:
            questionindex[array[1]] = clusterindex
            clusterinfo[clusterindex] = clusterinfo.get(clusterindex,[])
            clusterinfo[clusterindex].append(array[1])
            clusterindex += 1

        if questionindex.get(array[2],-1)==-1:
            if array[0] == '1':
                questionindex[array[2]] = questionindex[array[1]]
                clusterinfo[questionindex[array[1]]].append(array[2])
            else:
                questionindex[array[2]] = clusterindex
                clusterinfo[clusterindex] = clusterinfo.get(clusterindex, [])
                clusterinfo[clusterindex].append(array[2])
                clusterindex += 1

        if array[0] == '0':
            assert questionindex[array[2]] != questionindex[array[1]]
        else:
            if questionindex[array[2]] != questionindex[array[1]]:
                clusterindexsim[questionindex[array[2]]] = clusterindexsim.get(questionindex[array[2]],[])
                clusterindexsim[questionindex[array[2]]].append(questionindex[array[1]])
                clusterindexsim[questionindex[array[1]]] = clusterindexsim.get(questionindex[array[1]], [])
                clusterindexsim[questionindex[array[1]]].append(questionindex[array[2]])
    count+=1

f = open('../input/questionclusterdebug','w')
# f = open('./debug','w')
index2 = set()
count = 0
questionindex = {}
clusterinfo2 = {}
for k,v in clusterinfo.items():
    ids = set()
    res = set()
    index1 = set([k])
    while len(index1 - index2) > 0:
        temp = set()
        for k2 in index1 - index2:
            ids.add(k2)
            index2.add(k2)
            res = res | set(clusterinfo[k2])
            # for k3 in clusterindexsim.get(k2,[]):
            temp = temp | set(clusterindexsim.get(k2,[]))
        index1 = index1 | temp

    if len(res) > 0:
        print >> f,str(count) + "|" + ",".join(list(res))
        clusterinfo2[count] = ",".join(list(res))
        for question in res:
            questionindex[question] = count
        count += 1
# print s

count = 0
clusternotconnect = {}
pairsintrain = set()
for line in open('../input/train.csv'):
    if not line:
        continue
    array = line.strip().split(',')
    if array[0] == 'label':
        continue
    if count < limit:
        if array[0] == '0':
            clusternotconnect[questionindex[array[2]]] = clusternotconnect.get(questionindex[array[2]],set())
            clusternotconnect[questionindex[array[2]]].add(questionindex[array[1]])
            clusternotconnect[questionindex[array[1]]] = clusternotconnect.get(questionindex[array[1]],set())
            clusternotconnect[questionindex[array[1]]].add(questionindex[array[2]])
        pairsintrain.add(array[1] + array[2])
        pairsintrain.add(array[2] + array[1])
    count += 1

f = open('../input/questioncluster','w')
for k,v in clusterinfo2.items():
    print >> f, str(k) + "|" + v + "|" + ",".join(map(str, list(clusternotconnect.get(k,set()))))  # clusterid|questions|sameid|falseid
f.close()
import random
import itertools
clusterindexlist = []
questionindex = {}
clusterinfo = {}
allquestion = set()
clusternotconnect = {}
count = 0
for line in open('../input/questioncluster'):
    if not line:
        continue
    array = line.strip().split('|')
    id = array[0]
    questions = array[1].split(',')
    clusterinfo[id] = questions
    clusterindexlist.append(id)
    for question in questions:
        allquestion.add(question)
        questionindex[question] = id
        clusternotconnect[id] = array[2].split(',')

questioncountpos = {}
questioncountneg = {}
countt = 0
countf = 0
print len(allquestion)
print len(filter(lambda x:len(x[1]) > 2,clusterinfo.items()))
f2 = open('../input/train_remix.csv','w')
print >>f2,"label,q1,q2"
for id,l in clusterinfo.items():
    limit = 0
    res = []
    if len(l) > 2:
        otherquestion = list(allquestion - set(l))
        comb = [w for w in itertools.combinations(l, 2)]
        limit = min(int(len(l) * 8),max(1,len(l) * (len(l) - 1) / 3))
        res = random.sample(comb, min(limit+10,max(1,len(l) * (len(l) - 1) / 3)))
        count = 0
        for k in res:
            if k[0] + k[1] not in pairsintrain:
                print >> f2, "1," + k[0] + "," + k[1]
                count += 1
                countt += 1
            # else:
            #     continue
            if random.random() > 0.4:
                q1 = k[0]
                q2 = random.sample(otherquestion,1)
                if q2[0] + q1 not in pairsintrain:
                    print >> f2, "0," + q2[0] + "," + q1
                    countf+=1
            if random.random() > 0.4:
                q1 = k[1]
                q2 = random.sample(otherquestion,1)
                if q1 + q2[0] not in pairsintrain:
                    print >> f2, "0," + q1 + "," + q2[0]
                    countf+=1
            if count == limit:
                break
print countt,countf
f2.close()

f = open('../input/test_fix.csv', 'w')
print >> f, "q1,q2,oldindex"
count = 0
singlecount = 0
twocount = 0
fixcount = 0
for line in open('../input/test.csv'):
    if not line:
        continue
    array = line.strip().split(',')
    if array[0] == 'q1':
        continue
    # label = array[0]
    q2 = array[1]
    q1 = array[0]
    q1l = []
    q2l = []
    if questionindex.get(q1,-1) != -1:
        q1l = random.sample(clusterinfo[questionindex[q1]],min(len(clusterinfo[questionindex[q1]]),10))
    if questionindex.get(q2,-1) != -1:
        q2l = random.sample(clusterinfo[questionindex[q2]],min(len(clusterinfo[questionindex[q2]]),10))
    if len(q1l) <= 1:
        singlecount += 1
    if len(q2l) <= 1:
        singlecount += 1
    q1l = set(q1l) - set([q1])
    q2l = set(q2l) - set([q2])
    print >>f,  q1 + ',' + q2 + "," + str(count)
    for k2 in random.sample(q2l,min(len(q2l),10)):
        if random.random() > 0.5:
            print >>f,  q1 + ',' + k2 + "," + str(count)
        else:
            print >> f,k2 + ',' + q1 + "," + str(count)
    for k1 in random.sample(q1l,min(len(q1l),10)):
        if random.random() > 0.5:
            print >>f,  k1 + ',' + q2 + "," + str(count)
        else:
            print >>f,  q2 + ',' + k1 + "," + str(count)
    count += 1