#函数定义训练
#用getNum获取用户输入，用mean函数计算均值，用dev函数计算标准差，用median函数计算中位数
#提示：获取列表a长度len(a)；m的n次方使用函数pow(m,n)计算；对列表a内元素从小到大排序为a.sort()
import math

def getNum():       #获取用户不定长度的输入
    nums=input("请输入一组数字,用空格隔开： ")
    ls=[float(x) for x in nums.split()]
    return ls

def mean(numbers):  #计算平均值
    return sum(numbers)/len(numbers)
    
def dev(numbers, mean): #计算标准差
    sdev = 0.0
    variance = sum([(x-mean)**2 for x in numbers])/len(numbers)
    sdev = math.sqrt(variance)
    return sdev

def median(numbers):    #计算中位数
    numbers.sort()
    if len(numbers)%2==0:
        return (numbers[len(numbers)//2]+numbers[len(numbers)//2-1])/2
    else:
        return numbers[len(numbers)//2]#//返回一个整形，而/返回一个flaot形，list不能使用
    
n =  getNum() #主体函数
m =  mean(n)
print("平均值:{:.2f},标准差:{:.2f},中位数:{}".format(m,dev(n,m),median(n)))
