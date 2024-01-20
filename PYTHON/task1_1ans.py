k=eval(input())
sum=0
for x in range(1,k+1):
    d=(1+1/x)**x
    sum+=d
print("{:.8f}".format(sum))
