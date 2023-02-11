import numpy as np
vals = np.array([2,5,2,5,3,0,5,2,6,3,2,5,1,3,4])
print("Initial array: ", vals)
sort_index = np.argsort(vals,kind="stable") # Sorted in ascending order (får indexer i økende rekkefølge fra starten)
print("Indicies: ", sort_index)
sorted_val=np.sort(vals, kind="stable") # Sorted in ascending order
print("Sorted array: ", sorted_val)
