
File format for instances:

The first two lines:

N: number of nodes (excluding the depot)
M: number of clusters (excluding the depot)

The next M lines: M cluster sets

The first number of each line: the number of nodes in the corresponding cluster
The rest of numbers of each line: the indices of nodes that are in the corresponding cluster

The next N+1 lines below line Distance Matrix: the distance matrix (including the depot)

The next N+1 lines below line Traveling Time Matrix: the traveling time matrix (including the depot)

The next N+1 lines below line Time Windows: time windows for N+1 nodes (including the depot)

