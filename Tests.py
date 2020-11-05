import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("clean_loan.csv")

for item in data:
    data.hist(column=item)
    plt.savefig('C:/Users/Saulo/source/repos/Classification Project/Figures/'+item+'_distribution.png')
    plt.close()