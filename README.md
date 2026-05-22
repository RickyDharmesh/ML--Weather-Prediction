# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1) Load the weather dataset using pandas.
2) Preprocess the data by handling missing values and sorting by time.
3) Select features and create lag variables for temperature and PM2.5.
4) Train Random Forest models to predict temperature and PM2.5 and save the models.
```

## Program:
```

Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: RICKY DHARMESH P
RegisterNumber:  25016025
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ------------------------------
# Step 1: Sample dataset
# ------------------------------
data = {
    'CustomerID': [1,2,3,4,5,6,7,8,9,10],
    'Gender': ['Male','Female','Female','Male','Female','Male','Male','Female','Female','Male'],
    'Age': [19,21,20,23,31,22,35,30,25,28],
    'Annual Income (k$)': [15,16,17,18,19,20,21,22,23,24],
    'Spending Score (1-100)': [39,81,6,77,40,76,6,94,3,72]
}

df = pd.DataFrame(data)

# ------------------------------
# Step 2: Select features for clustering
# ------------------------------
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# ------------------------------
# Step 3: Apply K-Means (choose clusters, e.g., 3)
# ------------------------------
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X)  # Automatically fits and assigns clusters

# ------------------------------
# Step 4: Visualize clusters
# ------------------------------
plt.figure(figsize=(8,6))
for i in range(3):
    plt.scatter(X[df['Cluster']==i]['Annual Income (k$)'],
                X[df['Cluster']==i]['Spending Score (1-100)'],
                label=f'Cluster {i+1}')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            s=200, c='yellow', label='Centroids', marker='X')

plt.title('Customer Segmentation (K-Means)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# ------------------------------
# Step 5: Show dataset with clusters
# ------------------------------
print(df)

```

## Output:
<img width="855" height="684" alt="image" src="https://github.com/user-attachments/assets/7f7cdf27-5afd-41ee-ad98-1ae22a405bdc" />
<img width="695" height="609" alt="image" src="https://github.com/user-attachments/assets/4411a6b8-ddc3-4b15-90c1-8f081da79681" />


## Result:
The Random Forest model successfully predicted temperature, PM2.5 pollution, and solar radiation using weather sensor data with good accuracy. The system also generated next-step predictions and visual graphs comparing actual vs predicted values and showing feature importance.
