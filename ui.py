#! /usr/bin/env python3

# Original Authors  : Tony DiCola (tony@tonydicola.com)
#                   : William(Based on Adrian Rosebrock's OpenCV code on pyimagesearch.com)

import sys
import os
import argparse

from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306
from rpi_ws281x import *
import logging
import time
import threading
import queue
import subprocess
import psutil
import select
import tty
import termios
import pyudev
import evdev
import functools
import RPi.GPIO as GPIO
import numpy as np

COLOR_RED   = (0xFF, 0x00, 0x00)
COLOR_ORANGE= (0xFF, 0xA5, 0x00)
COLOR_YELLOW= (0xFF, 0xFF, 0x00)
COLOR_GREEN = (0x00, 0xFF, 0x00)
COLOR_BLUE  = (0x00, 0x00, 0xFF)
COLOR_TEAL  = (0x00, 0x80, 0x80)
COLOR_BGREEN= (0x0D, 0x98, 0xBA)
COLOR_BLACK = (0x00, 0x00, 0x00)

com = None
device = None
try:
    com = i2c(port = 1, address=0x3C)
    device = ssd1306(com, rotate = 0)
except:
    print ("OLED Disconnected")


JS_MIN_MOVE = 0.20
JOYSTICK_NAMES = ["8BitDo SN30 Pro+"]
#ENABLE_MOTORS = False
ENABLE_MOTORS = True


#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

NAME = os.path.basename(os.path.realpath(__file__))

DESCRIPTION = "\n" \
              "\n" \
              "usage: %s [options]\n" % NAME

EPILOG = "\n" \
         "\n" \
         "Examples:\n" \
         "\tSomething\n" \
         "\n"


class MotorControl:
    MOTOR_A_EN    = 4
    MOTOR_B_EN    = 17

    MOTOR_A_PIN1  = 26
    MOTOR_A_PIN2  = 21
    MOTOR_B_PIN1  = 27
    MOTOR_B_PIN2  = 18

    DIR_FORWARD   = 0
    DIR_BACKWARD  = 1

    left_forward  = 1
    left_backward = 0

    right_forward = 0
    right_backward= 1

    pwn_A = 0
    pwm_B = 0

    def __init__(self):
        global pwm_A, pwm_B
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.MOTOR_A_EN, GPIO.OUT)
        GPIO.setup(self.MOTOR_B_EN, GPIO.OUT)
        GPIO.setup(self.MOTOR_A_PIN1, GPIO.OUT)
        GPIO.setup(self.MOTOR_A_PIN2, GPIO.OUT)
        GPIO.setup(self.MOTOR_B_PIN1, GPIO.OUT)
        GPIO.setup(self.MOTOR_B_PIN2, GPIO.OUT)

        self.stop()
        try:
            pwm_A = GPIO.PWM(self.MOTOR_A_EN, 1000)
            pwm_B = GPIO.PWM(self.MOTOR_B_EN, 1000)
        except:
            pass

    def __del__(self):
        self.stop()
        GPIO.cleanup()

    def move (self, lmotor, rmotor):
        pass
         
    def motor_right(self, status, direction, speed):#Motor 2 positive and negative rotation
        if status == 0: # stop
            GPIO.output(self.MOTOR_B_PIN1, GPIO.LOW)
            GPIO.output(self.MOTOR_B_PIN2, GPIO.LOW)
            GPIO.output(self.MOTOR_B_EN, GPIO.LOW)
        else:
            if direction == self.DIR_FORWARD:
                GPIO.output(self.MOTOR_B_PIN1, GPIO.HIGH)
                GPIO.output(self.MOTOR_B_PIN2, GPIO.LOW)
                pwm_B.start(100)
                pwm_B.ChangeDutyCycle(speed)
            elif direction == self.DIR_BACKWARD:
                GPIO.output(self.MOTOR_B_PIN1, GPIO.LOW)
                GPIO.output(self.MOTOR_B_PIN2, GPIO.HIGH)
                pwm_B.start(0)
                pwm_B.ChangeDutyCycle(speed)


    def motor_left(self, status, direction, speed):#Motor 1 positive and negative rotation
        if status == 0: # stop
            GPIO.output(self.MOTOR_A_PIN1, GPIO.LOW)
            GPIO.output(self.MOTOR_A_PIN2, GPIO.LOW)
            GPIO.output(self.MOTOR_A_EN, GPIO.LOW)
        else:
            if direction == self.DIR_FORWARD:#
            #if direction == self.DIR_BACKWARD:#
                GPIO.output(self.MOTOR_A_PIN1, GPIO.HIGH)
                GPIO.output(self.MOTOR_A_PIN2, GPIO.LOW)
                pwm_A.start(100)
                pwm_A.ChangeDutyCycle(speed)
            elif direction == self.DIR_BACKWARD:
            #elif direction == self.DIR_FORWARD:
                GPIO.output(self.MOTOR_A_PIN1, GPIO.LOW)
                GPIO.output(self.MOTOR_A_PIN2, GPIO.HIGH)
                pwm_A.start(0)
                pwm_A.ChangeDutyCycle(speed)


    def stop(self):
        GPIO.output(self.MOTOR_A_PIN1, GPIO.LOW)
        GPIO.output(self.MOTOR_A_PIN2, GPIO.LOW)
        GPIO.output(self.MOTOR_B_PIN1, GPIO.LOW)
        GPIO.output(self.MOTOR_B_PIN2, GPIO.LOW)
        GPIO.output(self.MOTOR_A_EN,   GPIO.LOW)
        GPIO.output(self.MOTOR_B_EN,   GPIO.LOW)

class LED:
    def __init__(self):
        self.LED_COUNT      = 16      # Number of LED pixels.
        self.LED_PIN        = 12      # GPIO pin connected to the pixels (18 uses PWM!).
        self.LED_FREQ_HZ    = 800000  # LED signal frequency in hertz (usually 800khz)
        self.LED_DMA        = 10      # DMA channel to use for generating signal (try 10)
        self.LED_BRIGHTNESS = 255     # Set to 0 for darkest and 255 for brightest
        self.LED_INVERT     = False   # True to invert the signal (when using NPN transistor level shift)
        self.LED_CHANNEL    = 0       # set to '1' for GPIOs 13, 19, 41, 45 or 53

        # Create NeoPixel object with appropriate configuration.
        self.strip = Adafruit_NeoPixel( self.LED_COUNT,
                                        self.LED_PIN,
                                        self.LED_FREQ_HZ,
                                        self.LED_DMA,
                                        self.LED_INVERT,
                                        self.LED_BRIGHTNESS,
                                        self.LED_CHANNEL)
        # Intialize the library (must be called once before other functions).
        self.strip.begin()

    # Define functions which animate LEDs in various ways.
    def color_wipe(self, R, G, B):
        """Wipe color across display a pixel at a time."""
        color = Color(R,G,B)
        for i in range(self.strip.numPixels()):
            self.strip.setPixelColor(i, color)
            self.strip.show()

class OLED(threading.Thread):

    text_1 = "NO IP"
    text_2 = ""
    text_3 = ""
    text_4 = ""
    text_5 = ""
    text_6 = ""



    def __init__(self, *args, **kwargs):
        super(OLED, self).__init__(*args, **kwargs)
        self.__flag = threading.Event()  # 用于暂停线程的标识
        self.__flag.set()    # 设置为True
        self.__running = threading.Event()    # 用于停止线程的标识
        self.__running.set()    # 将running设置为True

    def run(self):
        while self.__running.isSet():
            self.__flag.wait()    # 为True时立即返回, 为False时阻塞直到内部的标识位为True后返回
            with canvas(device) as draw:
                draw.text((0,  0), self.text_1, fill="white")
                draw.text((0, 10), self.text_2, fill="white")
                draw.text((0, 20), self.text_3, fill="white")
                draw.text((0, 30), self.text_4, fill="white")
                draw.text((0, 40), self.text_5, fill="white")
                draw.text((0, 50), self.text_6, fill="white")
            #print('loop')
            self.pause()

    def pause(self):
        self.__flag.clear()  # 设置为False, 让线程阻塞

    def resume(self):
        self.__flag.set() # 设置为True, 让线程停止阻塞

    def stop(self):
        self.__flag.set()    # 将线程从暂停状态恢复, 如何已经暂停的话
        self.__running.clear()    # 设置为False

    def screen_show(self, position, text):
        if position == 1:
            self.text_1 = text
        elif position == 2:
            self.text_2 = text
        elif position == 3:
            self.text_3 = text
        elif position == 4:
            self.text_4 = text
        elif position == 5:
            self.text_5 = text
        elif position == 6:
            self.text_6 = text
        self.resume()

def open_evdev():
    evdev_devices = evdev.list_devices()
    joystick = None
    for dpath in evdev_devices:
        try:
            joystick = evdev.InputDevice(dpath)
            logging.debug("open_evdev: Checking name: %s" % joystick.name)
            if joystick.name in JOYSTICK_NAMES:
                logging.info("open_evdev: Found %s" % joystick.name)
                break
            del (joystick)
            joystick = None

        except OSError as err:
            logging.debug("open_evdev: Failed ot open Evdev Device: %s". str(err))
        except FileNotFoundError as err:
            logging.debug("open_evdev: File not found: %s" % str(err))
    return joystick



def color_thread(led, led_queue):
    while True:
        v = led_queue.get(block=True)
        R = v[0]
        G = v[1]
        B = v[2]
        led.color_wipe(R, G, B)

def move_thread_func(motor_control, move_queue):
    global ENABLE_MOTORS
    while True:
        v = move_queue.get(block=True)
        if v is None:
            logging.debug("Stop motor")
            motor_control.stop()
        else:
            rm_val = v[0]
            lm_val = v[1]
            logging.info("RM: %0.02f LM: %0.02f" % (rm_val, lm_val))

            rm_status = 0
            lm_status = 0


            rm_speed = int(np.abs(v[0] * 100))
            lm_speed = int(np.abs(v[1] * 100))

            if rm_speed > 0: rm_status = 1
            if lm_speed > 0: lm_status = 1

            rm_dir = motor_control.DIR_FORWARD
            if v[0] < 0: rm_dir = motor_control.DIR_BACKWARD

            lm_dir = motor_control.DIR_FORWARD
            if v[1] < 0: lm_dir = motor_control.DIR_BACKWARD

            logging.info("M S:DIR:SPEED %d:%d:% 3d | %d:%d:% 3d" % (rm_status, rm_dir, rm_speed, lm_status, lm_dir, lm_speed))
            #XXX: ENABLE THE MOTORS
            if ENABLE_MOTORS:
                motor_control.motor_right(rm_status, rm_dir, rm_speed)
                motor_control.motor_left(lm_status, lm_dir, lm_speed)

def subprocess_thread_func(command_queue):
    proc = None
    out_comm = None
    error_comm = None
    while True:
        v = command_queue.get(block = True)
        if proc is not None:
            logging.info("Killing Process")
            #if v[0] == None:
            proc.kill()
            proc
        else:
            logging.info("Command: %s" % command)
            proc = subprocess.Popen([script_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    #output, errors = proc.communicate(input = "Hello")
    #while (proc.poll() is None):
    #    print ("Waiting...")
    #print (output)
    #print(errors)


def main(argv):
    #Parse out the commandline arguments
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=DESCRIPTION,
        epilog=EPILOG
    )

    parser.add_argument("-t", "--test",
                        nargs=1,
                        default=["something"])

    parser.add_argument("-d", "--debug",
                        action="store_true",
                        help="Enable Debug Messages")

    args = parser.parse_args()
    #print ("Running Script: %s" % NAME)


    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        logging.debug("Openning NEO Pixels")
        led = LED()

        led_queue = queue.Queue(maxsize = 10)
        led_thread = threading.Thread(target=color_thread, args=(led, led_queue,), daemon=True)
        led_thread.start()
    except RuntimeError as err:
        logging.error ("Failed to open NEO Pixel: %s" % str(err))
        return

    logging.debug("Openning Motor Control")
    motor_control = MotorControl()
    move_queue = queue.Queue(maxsize = 1)
    move_thread = threading.Thread(target=move_thread_func, args=(motor_control, move_queue,), daemon=True)
    move_thread.start()

    logging.debug("Starting Subprocess Thread")
    command_queue = queue.Queue(maxsize = 1)
    command_thread = threading.Thread(target=subprocess_thread_func, args=(command_queue, ), daemon=True)
    command_thread.start()

    # Open the OLED Screen
    screen = OLED()
    screen.start()

    cmd = "hostname -I | cut -d\' \' -f1"
    IP = subprocess.check_output(cmd, shell=True).decode("utf-8")
    IP = IP.strip()
    time.sleep(0.5)
    #print ("%s" % str(IP))
    screen.screen_show(1, "IP: %s" % IP)

    joystick = open_evdev()

    context = pyudev.Context()
    monitor = pyudev.Monitor.from_netlink(context)
    monitor.filter_by(subsystem='input')
    monitor.start()

    running = True
    stdin_fn = sys.stdin.fileno()

    time.sleep(0.5)
    inputs = [sys.stdin.fileno(), monitor.fileno()]
    if joystick is None:
        screen.screen_show(2, "JS: Not Connected")
        led_queue.put(COLOR_ORANGE)
    else:
        screen.screen_show(2, "JS: %s" % joystick.name)
        inputs.append(joystick.fd)
        led_queue.put(COLOR_BGREEN)
    outputs = []
    exceptions = []

    try:
        #old_settings = termios.tcgetattr(stdin_fn)
        #tty.setraw(stdin_fn)

        my = 0.0
        mx = 0.0
        #ax = 0
        #ay = 0


        while running:
            r, w, x = select.select(inputs, outputs, exceptions)

            if sys.stdin.fileno() in r:
                #data = sys.stdin.read(1)
                data = sys.stdin.readline()
                logging.debug("STDIN Data: %s" % data)
                if data.strip() == 'q':
                    logging.debug("Quit")
                    running = False


            if joystick is not None and joystick.fd in r:
                try:
                    for event in joystick.read():
                        logging.debug (event)

                        if event.type == evdev.ecodes.EV_KEY:
                            if event.code == evdev.ecodes.BTN_TR:
                                print ("Finished")
                                running = False

                        #print ("Absolute Info: %s" % str(evdev.AbsInfo))
                        if event.type == evdev.ecodes.EV_ABS:
                            if event.code == evdev.ecodes.ABS_X:
                                abs_info = joystick.absinfo(evdev.ecodes.ABS_X)
                                mx =  1 * (event.value - ((abs_info.max + 1) / 2))
                                mx = mx / ((abs_info.max + 1) / 2)
                            if event.code == evdev.ecodes.ABS_Y:
                                abs_info = joystick.absinfo(evdev.ecodes.ABS_Y)
                                my = -1 * (event.value - ((abs_info.max + 1) / 2))
                                my = my / ((abs_info.max + 1) / 2)

                        abs_info = joystick.absinfo(evdev.ecodes.ABS_X)
                        mag = np.sqrt(my ** 2 + mx ** 2)
                        if mag > 1.0:
                            mag = 1.0
                        
                        rads = np.arctan2(my, mx)
                        degree = rads / np.pi * 180
                        logging.debug ("M: (x) %0.02f, (y) %0.02f: MAX: %d: Mag: %f, degree: %f" % (mx, my, abs_info.max, mag, degree))


                        # Motor Values
                        rm_val = 0.0
                        lm_val = 0.0
                        if (degree >=   0) and (degree <  90):
                            rm_rads = 2 * (rads - np.pi / 4)
                            rm_val = np.sin(rm_rads)
                            lm_val = 1.0
                        if (degree >=  90) and (degree < 180):
                            rm_val = 1.0
                            lm_rads = -2 * (rads - np.pi * 3 / 4)
                            lm_val = np.sin(lm_rads)
                        if (degree >= -180) and (degree < -90):
                            rm_rads = -2 * (rads - np.pi * 5 / 4)
                            rm_val = np.sin(rm_rads)
                            lm_val = -1.0
                        if (degree >= -90) and (degree < 0):
                            rm_val = -1.0
                            lm_rads = 2 * (rads - np.pi * 7 / 4)
                            lm_val = np.sin(lm_rads)


                        rm_val_scaled = rm_val * mag
                        lm_val_scaled = lm_val * mag
                        logging.debug ("RM (scaled): %0.02f (%0.02f), LM (scaled): %0.02f (%0.02f)" % (rm_val, rm_val_scaled, lm_val, lm_val_scaled))
                        if (np.abs(rm_val_scaled)) < JS_MIN_MOVE:
                            rm_val_scaled = 0.0
                        if (np.abs(lm_val_scaled)) < JS_MIN_MOVE:
                            lm_val_scaled = 0.0
                            
                        if move_queue.empty():
                            move_queue.put((rm_val_scaled, lm_val_scaled))

                except OSError as err:
                    logging.info("JS disconnected!")
                    inputs.remove(joystick.fd)

            if monitor.fileno() in r:
                for udev in iter(functools.partial(monitor.poll, 0), None):
                    if not udev.device_node:
                        break
                    if udev.action == u'add':
                        logging.debug("Added Input Device: %s" % udev)
                        if joystick is None:
                            joystick = open_evdev()

                        if joystick is None:
                            screen.screen_show(2, "JS: Not Connected")
                        else:
                            screen.screen_show(2, "JS: %s" % joystick.name)


                    if joystick is not None:
                        inputs.append(joystick.fd)

                    if udev.action == u'remove':
                        logging.debug("Remove Input Device: %s" % udev)
                        if joystick is not None:
                            inputs.remove(joystick.fd)
                        del(joystick)
                        joystick = open_evdev()

                        if joystick is None:
                            screen.screen_show(2, "JS: Not Connected")
                            led_queue.put(COLOR_ORANGE)
                        else:
                            screen.screen_show(2, "JS: %s" % joystick.name)

    finally:
        #termios.tcsetattr(stdin_fn, termios.TCSADRAIN, old_settings)
        pass

    led_queue.put(COLOR_BLACK)
    move_queue.put(None)
    screen.stop()


if __name__ == "__main__":
    main(sys.argv)

