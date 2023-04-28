L = [1, 2, 3]
sublist = [L[i:i+j] for i in range(0,len(L)) for j in range(1,len(L)-i+1)]
print(sublist)