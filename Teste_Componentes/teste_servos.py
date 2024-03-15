from gpiozero import Servo
from time import sleep

servo1 = Servo(12, min_pulse_width=0.5/1000, max_pulse_width=2.2/1000)
servo2 = Servo(13, min_pulse_width=0.5/1000, max_pulse_width=2.2/1000)

def controlar_servo(servo):
    servo.min()
    sleep(1)
    servo.mid()
    sleep(1)
    servo.max()
    sleep(1)

while True:
    controlar_servo(servo1)
    controlar_servo(servo2)

