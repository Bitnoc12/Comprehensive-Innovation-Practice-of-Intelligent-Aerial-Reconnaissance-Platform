# python基础综合训练
# 获得用户输入数字 n，计算并输出从 n 开始的 5 个质数，单行输出，质数间用逗号、分割。
# 注意：需要考虑用户输入的数字 N 可能是浮点数，应对输入取整数；最后一个输出后不用逗号
# 注意：需要对输入小数情况进行判断，获取超过该输入的最小整数
def is_prime(num):
    # 判断是否为质数
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

n = float(input())
N = int(n)
N = N + 1 if N < n else N
count = 5
primes = []

while count > 0:
    if is_prime(N):
        primes.append(str(N))
        count -= 1
    N += 1

output = ", ".join(primes)
print(output)
