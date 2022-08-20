file = open('percent.txt')
s = 0
next = file.readline()
weights = []
while next:
    next = float(next)
    weights.append(next)
    next = file.readline()
s = sum(weights)
for w in weights:
    print('{:0.1f}'.format(w / s * 100))
