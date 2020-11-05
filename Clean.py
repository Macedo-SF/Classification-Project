import pandas as pd
import numpy as np
import uszipcode
import sys
from collections import Counter
from uszipcode import SearchEngine
search = SearchEngine()

#functions ----------------------------------------------------------------------------------------------------

def myPrint(input):
    print('--------------------------------------------------')
    print(input)
    print('--------------------------------------------------')

#functions end ------------------------------------------------------------------------------------------

#load data ----------------------------------------------------------------------------------------------------

#write output to txt
sys.stdout=open('C:/Users/Saulo/source/repos/Classification-Project/clean_output.txt','w') #unzip before
                                                                                    #github wouldn't allow, too big
data = pd.read_csv("SBAnational.csv")
#size, indexes and data types
myPrint('Initial Shape and Data Type')
myPrint(data.shape)
myPrint(data.dtypes)
#checking for nulls
myPrint(data.isnull().sum())
#target before cleaning
myPrint('Initial Target Status')
myPrint(Counter(data.MIS_Status).values())
myPrint(Counter(data.MIS_Status).keys())
#MIS_Status is what we want to predict, it shows whether the loan was paid in full or 
    #charged off(the loan was deemed unlikely to be collected by the loaner)
        #some of the rows are null for MIS_Status, we'll drop them
        #city, state, newexist and lowdoc are promising features, we'll also drop their nulls
data = data.dropna(subset=['MIS_Status','City','State','NewExist','LowDoc'])

#some columns are useless --------------------------------------------------------------------------------
    #Name and LoanNr_ChkDgt are unique identifiers
    #zips repeat what city and state tells us
    #the banks and their locations should not influence
    #nor should dates, we can use them to explore, but no more
    #sbaappv and revline are irrelevant
    #disbursement and balance are post default
data = data.drop(columns=['Name','LoanNr_ChkDgt','Zip','Bank','BankState','ApprovalDate','ApprovalFY','SBA_Appv',
                          'DisbursementDate','DisbursementGross','ChgOffDate','BalanceGross','ChgOffPrinGr',
                          'RevLineCr','City'])

#here we'll further clean the data ----------------------------------------------------------------------

#City

#dropped for now
#myPrint('City Status')
#myPrint(data.City.describe())
#myPrint(Counter(data.City).values())
#myPrint(Counter(data.City).keys())

#State

myPrint('State Status')
states = {'IN':1, 'OK':2, 'FL':3, 'CT':4, 'NJ':5, 'NC':6, 'IL':7, 'RI':8, 'TX':9, 'VA':10,
        'TN':11, 'AR':12, 'MN':13, 'MO':14, 'MA':15, 'CA':16, 'SC':17, 'LA':18, 'IA':19, 'OH':20,
        'KY':21, 'MS':22, 'NY':23, 'MD':24, 'PA':25, 'OR':26, 'ME':27, 'KS':28, 'MI':29, 'AK':30,
        'WA':31, 'CO':32, 'MT':33, 'WY':34, 'UT':35, 'NH':36, 'WV':37, 'ID':38, 'AZ':39, 'NV':40,
        'WI':41, 'NM':42, 'GA':43, 'ND':44, 'VT':45, 'AL':46, 'NE':47, 'SD':48, 'HI':49, 'DE':50,
        'DC':51}
data.State = [states[item] for item in data.State]
myPrint(data.State.describe())
myPrint(Counter(data.State).values())
myPrint(Counter(data.State).keys())

#NAICS

data.NAICS = [str(item)[:2] for item in data.NAICS]
#code = {0:np.NaN}
data.NAICS = [int(item) for item in data.NAICS]
myPrint('NAICS Status')
myPrint(data.NAICS.describe())
myPrint(Counter(data.NAICS).values())
myPrint(Counter(data.NAICS).keys())

#NewExist

neTAG = {1:0,2:1,0:np.NaN}
data.NewExist = [neTAG[item] for item in data.NewExist]
data = data.dropna(subset=['NewExist'])
myPrint('NewExist Status')
myPrint(data.NewExist.describe())
myPrint(Counter(data.NewExist).values())
myPrint(Counter(data.NewExist).keys())

#Term

myPrint('Term Status')
myPrint(data.Term.describe())
myPrint(Counter(data.Term).values())
myPrint(Counter(data.Term).keys())

#NoEmp

myPrint('NoEmp Status')
myPrint(data.NoEmp.describe())
myPrint(Counter(data.NoEmp).values())
myPrint(Counter(data.NoEmp).keys())

#CreateJob

myPrint('CreateJob Status')
myPrint(data.CreateJob.describe())
myPrint(Counter(data.CreateJob).values())
myPrint(Counter(data.CreateJob).keys())

#RetainedJob

myPrint('RetainedJob Status')
myPrint(data.RetainedJob.describe())
myPrint(Counter(data.RetainedJob).values())
myPrint(Counter(data.RetainedJob).keys())

#isFranchise

def isFranchise(input):
    if(input==0 or input==1):
        return 1
    else:
        return 0

data = data.rename(columns={'FranchiseCode':'isFranchise'},index={'FranchiseCode':'isFranchise'})
data.isFranchise = [isFranchise(item) for item in data.isFranchise]
myPrint('isFranchise Status')
myPrint(data.isFranchise.describe())
myPrint(Counter(data.isFranchise).values())
myPrint(Counter(data.isFranchise).keys())

#UrbanRural

data.UrbanRural = [int(item) for item in data.UrbanRural]
myPrint('UrbanRural Status')
myPrint(data.UrbanRural.describe())
myPrint(Counter(data.UrbanRural).values())
myPrint(Counter(data.UrbanRural).keys())

#LowDoc

ld = {'Y':1,'N':0, 'C':np.NaN,'1':np.NaN,'S':np.NaN,'R':np.NaN,'A':np.NaN,'0':np.NaN}
data.LowDoc = [ld[item] for item in data.LowDoc]
data = data.dropna(subset=['LowDoc'])
myPrint('LowDoc Status')
myPrint(data.LowDoc.describe())
myPrint(Counter(data.LowDoc).values())
myPrint(Counter(data.LowDoc).keys())

#MIS_Status

status = {'P I F': 1, 'CHGOFF': 0}
data.MIS_Status = [status[item] for item in data.MIS_Status]
myPrint(data.MIS_Status.describe()) #here, we can see it is unbalanced
myPrint('MIS_Status')
myPrint(Counter(data.MIS_Status).values())
myPrint(Counter(data.MIS_Status).keys())

#GrAppv

data.GrAppv = data.GrAppv.replace('[\$,]', '', regex=True).astype(float)
myPrint('GrAppv Status')
myPrint(data.GrAppv.describe())
#myPrint(Counter(data.GrAppv).values())
#myPrint(Counter(data.GrAppv).keys())

#target after cleaning
myPrint('Final Target Status')
myPrint(Counter(data.MIS_Status).values())
myPrint(Counter(data.MIS_Status).keys())
#data after clean up
myPrint('Final Data Status')
myPrint(data.isnull().sum())
myPrint(data.shape)
myPrint(data.dtypes)
#close output.txt
sys.stdout.close()
#export to new csv
data.to_csv(r'C:\Users\Saulo\source\repos\Classification-Project\clean_loan.csv',index=False)
#to do
    #use zip to fill city and state before dropping
    #remove all prints for unique values(like city and grant approval)
    #perhaps sort rows according to target value(needed for proper selection)
    #group up some features, terms in years, no emp in 10s, create job in 10s or 5s, retainedjob
        #some features have one too many keys but only a few carry many values