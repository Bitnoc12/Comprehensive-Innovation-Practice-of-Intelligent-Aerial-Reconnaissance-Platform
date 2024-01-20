#条件语句练习
#[0,60)差,(70,80]中,(80,90]良,(90,100]优
#根据成绩输出档次

while True:
    a=eval(input("输入成绩:"))

    if a>90 and a<=100:
        print("优")
    elif a>80 and a<=90:
        print("良")
    elif a>70 and a<=80:
        print("中")
    else:
        print("差")
    


