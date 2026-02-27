import numpy as np

# 6 cities as nodes in a graph, with 1/distances as edge weights
cities = ["Karachi", "Islamabad", "Swat", "Lahore", "Quetta", "Peshawar"]
A = np.array([
  [0, 1/1000, 0, 0, 0, 0],
  [1/1000, 0, 1/300, 1/400, 0, 1/300],
  [0, 1/300, 0, 0, 0, 1/250],
  [1/500, 0, 0, 0, 0, 0],
  [0, 0, 0, 1/600, 0, 0],
  [0, 0, 1/250, 0, 1/700, 0]
])

# symmetric matrix of distances between cities
B = A.T @ A

eigenvalues, eigenvectors = np.linalg.eigh(B)

principal_eigenvector = eigenvectors[:, -1]

# Index of the city with the highest centrality
hub_index = np.argmax(principal_eigenvector)
hub_city = cities[hub_index]

print(f"""Based on our analysis, {hub_city} is the ideal hub because it has 
the highest Network Reach ({principal_eigenvector[hub_index]}).
This means that {hub_city} is the most efficient 'transfer point' in 
the system. Because it is the easiest city to reach from everywhere 
else, it will naturally handle the highest volume of passengers and 
serves as the best location for our main refueling and maintenance base.""")