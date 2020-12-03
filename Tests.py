import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from statsmodels.stats.diagnostic import normal_ad

data = pd.read_csv("encoded_loan.csv")

for item in data:
    data.hist(column=item)
    plt.savefig('C:/Users/Saulo/source/repos/Classification-Project/Figures/Distribution/'+item+'_distribution.png')
    plt.close()
    #print(normal_ad(data[item]),'\n')

#correlation
dataC=data.drop(columns=['MIS_Status'])
f=plt.figure(figsize=(19, 15))
plt.matshow(dataC.corr(),fignum=f.number)
plt.xticks(range(dataC.shape[1]), dataC.columns, fontsize=14, rotation=45)
plt.yticks(range(dataC.shape[1]), dataC.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)
plt.savefig('C:/Users/Saulo/source/repos/Classification-Project/Figures/Distribution/correlation_matrix.png')
plt.close()