'''
Simple python script to get Asyncronous gamepad inputs
Thomas FLAYOLS - LAAS CNRS
From https://github.com/thomasfla/solopython

Use:
To display data, run "python gamepadClient.py"
'''
import inputs
import time
from multiprocessing import Process
from multiprocessing.sharedctypes import Value
from ctypes import c_double, c_bool


class KeyboardClient():
    def __init__(self):
        self.running = Value(c_bool, lock=True)
        self.vx = Value(c_double, lock=True)
        self.vy = Value(c_double, lock=True)

        self.vx.value = 0.0
        self.vy.value = 0.0
        args = (self.running, self.vx, self.vy)
        self.process = Process(target=self.run, args=args)
        self.process.start()
        time.sleep(0.2)

    def run(self, running, vx, vy):
        running.value = True
        while(running.value):
            events = inputs.get_key()
            for event in events:
                print(event.ev_type, event.code, event.state)
                if (event.ev_type == 'Key'):
                    if event.code == 'KEY_UP':
                        vx.value += 0.01
                    elif event.code == 'KEY_DOWN':
                        vx.value -= 0.01
                    elif event.code == 'KEY_RIGHT':
                        vy.value -= 0.005
                    elif event.code == 'KEY_LEFT':
                        vy.value += 0.005

    def stop(self):
        self.running.value = False
        self.process.terminate()
        self.process.join()
