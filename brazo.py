# Simple demo of of the PCA9685 PWM servo/LED controller library.
# This will move channel 0 from min to max position repeatedly.
# Author: Tony DiCola
# License: Public Domain
#from _future_ import division
import time
# Import the PCA9685 module.
# import Adafruit_PCA9685
from adafruit_servokit import ServoKit
pinza=False
kit = ServoKit(channels=16)
# Uncomment to enable debug output.
# import logging
# logging.basicConfig(level=logging.DEBUG)
# Initialise the PCA9685 using the default address (0x40).
# pwm = Adafruit_PCA9685.PCA9685()

# Alternatively specify a different address and/or bus:
# pwm = Adafruit_PCA9685.PCA9685(address=0x41, busnum=2)
# Configure min and max servo pulse lengths
servo_min = 200  # Min pulse length out of 4096
servo_max = 550  # Max pulse length out of 4096


# Helper function to make setting a servo pulse width simpler.
def set_servo_pulse(channel, pulse):
    pulse_length = 1000000  # 1,000,000 us per second
    pulse_length //= 60  # 60 Hz
    print('{0}us per period'.format(pulse_length))
    pulse_length //= 4096  # 12 bits of resolution
    print('{0}us per bit'.format(pulse_length))
    pulse *= 1000
    pulse //= pulse_length


# pwm.set_pwm(0,0,100)
# Set frequency to 60hz, good for servos.
# pwm.set_pwm_freq(60)
pos_m0 = 100
pos_m1 = 100
pos_m2 = 30
pos_m3 = 0
print('Moving servo on channel 0, press Ctrl-C to quit...')
i = 200


def moure_motor(motor, pos_motor, pos):
    i = 0
    if (pos_motor < pos):
        j = 1
    else:
        j = -1
    while (pos_motor != pos):
        i += 1
        pos_motor += j
        if (i == 5):
            i = 0
            print(pos_motor)
            kit.servo[motor].angle = pos_motor
            time.sleep(0.1)

    pos_m0 = pos_motor


# kit.servo[0].angle=180
# kit.servo[0].actuation_range=400
kit.servo[0].set_pulse_width_range(1000, 2150)
# kit.servo[2].set_pulse_width_range(1000,4000)
def jugada_1():
      # kit.servo[0].angle=50
    pos_inicial()
    abrir_pinza()
    pos_m0=110
    pos_m1=110
    pos_m2=0
    pos_m3=100
    time.sleep(2)
    pos_m3=100
    moure_motor(2, pos_m2, 40)
    moure_motor(1, pos_m1, 0)
    cerrar_pinza()
    pos_m1=0
    pos_m2=40    
    pos_inicial()
    pos_m0=110
    pos_m1=110
    pos_m2=0
    pos_m3=100
    time.sleep(2)
    pos_m3=100
    moure_motor(2, pos_m2, 40)
    moure_motor(1, pos_m1, 40)
    abrir_pinza()
    pos_m1=40
    pos_m2=40
    
    
    
    time.sleep(2)
def pos_inicial():
    moure_motor(1, pos_m1, 110)
    moure_motor(2, pos_m2, 0)
    moure_motor(0, pos_m0, 105)
    
    

def abrir_pinza():
    moure_motor(4,20,120)
def cerrar_pinza():
    moure_motor(4,120,20)
    time.sleep(0.2)
    moure_motor(4,20,0)
while True:
    # kit.servo[0].angle=50

    pos_inicial()
    abrir_pinza()
    moure_motor(0,105,55)
    moure_motor(2,0,25)
    moure_motor(1,105,30)
    moure_motor(1,30,5)
    pos_m2=25
    pos_m1=5
    pos_m0=55
    cerrar_pinza()
    time.sleep(1)
    pos_inicial()
    moure_motor(0,105,125)
    time.sleep(1)
    moure_motor(2,0,40)
    time.sleep(1)
    moure_motor(1,105,50)
    time.sleep(1)
    abrir_pinza()
    time.sleep(1)
    pos_m1=50
    pos_m2=40
    pos_m0=125
    '''matar pieza
    pos_inicial()
    abrir_pinza()
    pos_m0=110
    pos_m1=110
    pos_m2=0
    pos_m3=100
    time.sleep(2)
    pos_m3=100
    moure_motor(2, pos_m2, 45)
    moure_motor(1, pos_m1, 50)
    cerrar_pinza()
    pos_m2=45
    pos_m1=50
    pos_inicial()
    moure_motor(0, pos_m0, 180)
    moure_motor(2, pos_m2, 40)
    moure_motor(1, pos_m1, 40)
    abrir_pinza()
    pos_m0=180
    pos_m1=40
    pos_m2=40
    
    pos_inicial()
    abrir_pinza()
    pos_m0=110
    pos_m1=110
    pos_m2=0
    pos_m3=100
    time.sleep(2)
    pos_m3=100
    moure_motor(2, pos_m2, 35)
    moure_motor(1, pos_m1, 40)
    cerrar_pinza()
    pos_m1=40
    pos_m2=35    
    pos_inicial()
    pos_m0=110
    pos_m1=110
    pos_m2=0
    pos_m3=100
    time.sleep(2)
    pos_m3=100
    moure_motor(2, pos_m2, 45)
    moure_motor(1, pos_m1, 50)
    abrir_pinza()
    pos_m1=50
    pos_m2=45
    time.sleep(2)'''
    
    
    # kit.servo[1].angle=50
    '''moure_motor(0, 180, 110)
    moure_motor(2, pos_m2, 50)
    moure_motor(1, pos_m1, 60)
    pos_m1 = 60
    pos_m2 = 50
    pos_m3 = 100
    moure_motor(4, pos_m3, 0)
    pos_m3 = 0
    time.sleep(2)
    moure_motor(1, pos_m1, 100)
    moure_motor(2, pos_m2, 0)
    moure_motor(0, 110, 180)
    moure_motor(1, 100, 50)
    moure_motor(2, 0, 50)
    moure_motor(4, pos_m3, 100)
    moure_motor(1, 50, 100)
    moure_motor(2, 50, 0)

    pos_m1 = 100
    pos_m2 = 00
    pos_m3 = 100'''

