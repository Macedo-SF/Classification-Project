import pandas as pd
import numpy as np
import sys
from collections import Counter
from statsmodels.stats.diagnostic import normal_ad

#functions ----------------------------------------------------------------------------------------------------

def myPrint(input):
    print('--------------------------------------------------')
    print(input)
    print('--------------------------------------------------')

#functions end ------------------------------------------------------------------------------------------

sys.stdout=open('C:/Users/Saulo/source/repos/Classification-Project/encoding_output.txt','w')

data = pd.read_csv("clean_loan.csv")
#State
out={}
for i in range(1,52):
    test=data[data['State']==i]['MIS_Status']
    out[i]=100*test.mean() #apply points based on mean instead of just mean
    #print('State: ',i)
    #print('Count: ',test.count())
    #print('Mean: ',test.mean(),'\n')
data.State=[out[item] for item in data.State]
myPrint(data.State.describe())
myPrint(Counter(data.State).values())
myPrint(Counter(data.State).keys())
#NAICS
out={}
aux={0:0, 11:1, 21:2, 22:3, 23:4, 31:5, 32:5, 33:5, 42:6, 44:7, 45:7, 48:8, 49:8,
    51:9, 52:10, 53:11, 54:12, 55:13, 56:14, 61:15, 62:16, 71:17, 72:18, 81:19, 92:20}
data.NAICS=[aux[item] for item in data.NAICS]
myPrint(Counter(data.NAICS).values())
myPrint(Counter(data.NAICS).keys())
for i in range(0,21):
    test=data[data['NAICS']==i]['MIS_Status']
    out[i]=100*test.mean() #
    #print('State: ',i)
    #print('Count: ',test.count())
    #print('Mean: ',test.mean(),'\n')
data.NAICS=[out[item] for item in data.NAICS]
myPrint(data.NAICS.describe())
myPrint(Counter(data.NAICS).values())
myPrint(Counter(data.NAICS).keys())
#
sys.stdout.close()
data.to_csv(r'C:\Users\Saulo\source\repos\Classification-Project\encoded_loan.csv',index=False)