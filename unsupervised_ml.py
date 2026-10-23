from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

customers = [
    [1000, 2],
    [1500, 3],
    [2000, 2],
    [5000, 6],
    [6000, 7],
    [5500, 5],
    [12000, 12],
    [15000, 14],
    [13000, 11],
    [3000, 4]

]   

scaler = StandardScaler()
scaled_customers = scaler.fit_transform(customers)

model = KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(scaled_customers)

labels = model.labels_
for i, label in enumerate(labels, start=1):
    print(f"Customer {i}: Cluster {label}")
print()
for cluster_id in range(3):
    count = list(labels).count(cluster_id)
    print(f"Cluster {cluster_id}: {count} customers")

import matplotlib.pyplot as plt 

spending = [customer[0] for customer in customers]
visits = [customer[1] for customer in customers] 


plt.scatter(spending, visits, c= model.labels_,cmap="viridis" , s=100)
plt.xlabel("spending")
plt.ylabel("visits")
plt.title("customer clusters")
plt.show()