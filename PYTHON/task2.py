from djitellopy import tello
from time import sleep

t = tello.Tello()

t.connect()

print(t.get_battery())  # 电量

t.takeoff()

t.send_rc_control(0, 0, 0, 0)  # 悬停

t.move_up(70)  # 上飞70cm
sleep(1)

t.send_rc_control(0, 350, 0, 0)  # 前进 350cm/s
sleep(1)

t.move_forward(300)  # 前进300cm
sleep(1)

t.move_back(300)  # 后退300cm
sleep(1)

t.flip_right()  # 向右翻转
sleep(1)

t.flip_left()  # 向左翻转
sleep(1)

"""
t.rotate_clockwise(90)
t.rotate_clockwise(270)
t.send_rc_control(0,0,0,0)
sleep(1)
t.move_left(100)    
sleep(1)
"""

t.send_rc_control(0, 0, 0, 0)
sleep(1)

t.land()
