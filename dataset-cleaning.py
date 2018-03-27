import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

features = ["date", "meantempm", "meandewptm", "meanpressurem", "maxhumidity", "minhumidity", "maxtempm",
            "mintempm", "maxdewptm", "mindewptm", "maxpressurem", "minpressurem", "precipm"]
dataframe = pd.read_csv('dataset.csv').set_index('date')

def derive_nth_day_feature(df, feature, N):
    rows = df.shape[0]
    nth_prior_meassurements = [None]*N + [df[feature][i-N] for i in range(N, rows)]
    col_name = "{}_{}".format(feature, N)
    df[col_name] = nth_prior_meassurements
    
for feature in features:
    if feature != 'date':
        for N in range(1, 4):
            derive_nth_day_feature(dataframe, feature, N)    
            
#dataframe.columns      

#Removing non relevent features to reduce the dataset            
to_remove = [feature 
             for feature in features 
             if feature not in ['meantempm', 'mintempm', 'maxtempm']]

to_keep = [col for col in dataframe.columns if col not in to_remove]

dataframe = dataframe[to_keep]
#dataframe.columns           

dataframe.info()

#Converting datatype of every column to float

dataframe= dataframe.apply(pd.to_numeric, errors='coerce')

# Call describe on df and transpose it due to the large number of columns
spread = dataframe.describe().T

# precalculate interquartile range for ease of use in next calculation
IQR = spread['75%'] - spread['25%']

# create an outliers column which is either 3 IQRs below the first quartile or
# 3 IQRs above the third quartile
spread['outliers'] = (spread['min']<(spread['25%']-(3*IQR)))|(spread['max'] > (spread['75%']+3*IQR))

# just display the features containing extreme outliers
spread.loc[spread.outliers,]


#Analyzing outliers

'''plt.rcParams['figure.figsize'] = [14, 8]
dataframe.maxhumidity_1.hist()
plt.title('Distribution of maxhumidity_1')
plt.xlabel('maxhumidity_1')
plt.show()

plt.rcParams['figure.figsize'] = [14, 8]
dataframe.maxpressurem_1.hist()
plt.title('Distribution of maxpressurem_1')
plt.xlabel('maxpressurem_1')
plt.show()'''

dataframe=dataframe.dropna()
dataframe.to_csv("dataset-cleaned.csv")
