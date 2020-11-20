import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import normal_ad

data = pd.read_csv("encoded_loan.csv")

for item in data:
    data.hist(column=item)
    plt.savefig('C:/Users/Saulo/source/repos/Classification-Project/Figures/'+item+'_distribution.png')
    plt.close()
    #print(normal_ad(data[item]),'\n')