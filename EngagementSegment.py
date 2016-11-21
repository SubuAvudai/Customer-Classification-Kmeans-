# -*- coding: utf-8 -*-
"""
Created on Tues Nov 15 15:18:14 2016

@author: Subu
"""
#Importing Necessary Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Reading from Input File
dateparse = lambda x: pd.datetime.strptime(x, '%d-%m-%Y')
dataset = pd.read_csv("Online_Sales.csv",parse_dates=['DateKey'],date_parser=dateparse,dayfirst=True)

#Procuring the cutomer wise Sales Amount
customer_wise_amountsum=pd.DataFrame(dataset.groupby(dataset['CustomerKey']).apply(lambda subf: subf['SalesAmount'].sum()))

#Procuring the Frequency of buying of each Customer
transactions=pd.DataFrame(dataset.iloc[:,[1,2]])
uniquetransactions=pd.DataFrame(transactions.drop_duplicates(subset='TransactionID'))
no_oftransac_per_cust=uniquetransactions.groupby('CustomerKey').count()

#Procuring the discount availed by each customer
customer_wise_discountavailed=pd.DataFrame(dataset.groupby(dataset['CustomerKey']).apply(lambda subf: (subf['UnitPrice']-subf['SalesAmount']).sum()))


#Procuring number of subcategories each customer has bought from
no_of_subcateg_per_cust=pd.DataFrame(dataset.iloc[:,[1,5]]).groupby(['CustomerKey'])['ProductSubcategoryName'].nunique().reset_index()
uniquesubcategories=pd.DataFrame(dataset.iloc[:,[5]].drop_duplicates(subset='ProductSubcategoryName'))
#blah=pd.DataFrame(dataset.iloc[:,[1,6]]).groupby(['CustomerKey'])['ProductCategoryName'].nunique().reset_index()


c= pd.DataFrame(dataset.groupby(['CustomerKey','ProductCategoryName']).apply(lambda subf: subf['SalesAmount'].sum()/subf['ProductCategoryName'].count()).reset_index())
d= pd.DataFrame(c.iloc[:,2])
c['Amount']=d.values
categorical_weightage_per_cust=pd.DataFrame(c.groupby(c['CustomerKey']).apply(lambda subf: subf['Amount'].sum()))


"""
datetransactions=pd.DataFrame(dataset.iloc[:,[0,2,1]]).drop_duplicates(subset='TransactionID')
datetransactions.sort(columns=['CustomerKey','DateKey'],inplace=True)
temp=np.array(datetransactions.iloc[:,[1,2]])
dates=(datetransactions.iloc[:,[0]])
#print(dates)
list3=np.array(no_oftransac_per_cust.iloc[:,0])
datesum=0
avg_time_between_trans=[]
for i in range(39073):
    if(i == 39072):
        break;
    if(temp[i][1]==temp[i+1][1]):
         datesum=datesum+(dates.iloc[i+1,:]-dates.iloc[i,:])#astype('timedelta64[d]')
    else:
        #avg_time_between_trans.append(float(datesum/list3[i]))
        print(datesum)
        datesum=0
        
#Procuring Unique list of CustomerKeys
uniquecustomers=pd.DataFrame(dataset.iloc[:,[1]].drop_duplicates(subset='CustomerKey'))
uniquecustomers.sort(columns='CustomerKey',inplace=True)
"""



#Creating the Features array to perform clustering on
list1=np.array(customer_wise_amountsum.iloc[:,0])
list2=np.array(no_oftransac_per_cust.iloc[:,0])
X=np.column_stack((list1,list2))

#Checking the ideal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

#Creating list with the Engagement Scores
list3=[]
for i in y_kmeans:
    if i == 0:
        list3.append("Low Engaged")
    elif i == 1:
        list3.append("Very Highly Engaged")
    elif i == 2:
        list3.append("Inactive")
    elif i == 3:
        list3.append("Highly Engaged")
    elif i == 4:
        list3.append("Engaged")
        

# Visualising the clusters
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 15, c = 'blue',label='Very Highly Engaged')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 15, c = 'cyan',label='Highly Engaged')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 15, c = 'magenta',label='Engaged')
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 15, c = 'red',label='Low Engaged')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 15, c = 'green',label='Inactive')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 15, c = 'yellow', label = 'Centroids')
plt.title('EngagementSegment Cluster')
plt.xlabel('Sales Amount')
plt.ylabel('Frequency of Buying')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

#Writing to Excel File
import csv
Finaldata= np.column_stack((uniquecustomers,list3))
fl = open('Finaldata.csv', 'w')
writer = csv.writer(fl)
writer.writerow(['CustomerKey', 'EngagementScore']) #if needed
for values in Finaldata:
    writer.writerow(values)
fl.close()  

   
