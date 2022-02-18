
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data= pd.read_csv('Countryclusters.csv',encoding='latin1' )
plt.scatter(data['longitude'],data['latitude'])
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()

x=data.iloc[:,1:3]
kmeans=KMeans(3)
kmeans.fit(x)
idaubechies_clusters=kmeans.fit_predict(x)
print(idaubechies_clusters)
data_with_cluster=data.copy()
data_with_cluster['Clusters']=idaubechies_clusters
plt.scatter(data_with_cluster['longitude'],data_with_cluster['latitude'],c=data_with_cluster['Clusters'],cmap='rainbow')
plt.show()

wcss=[]
for i in range(1,7):
    kmeans=KMeans(i)
    kmeans.fit(x) 
    wcss_iter=kmeans.inertia_
    wcss.append(wcss_iter)
    
number_clusters=range(1,7)
plt.plot(number_clusters,wcss)
plt.title('the elbow title')
plt.xlabel('number of clusters') 
plt.ylabel('wcss')
plt.show()
  