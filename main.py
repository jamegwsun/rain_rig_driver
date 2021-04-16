from time import time
from rain_rig_driver_lib import run_motor


with run_motor() as m:
    m.RunMotorA(100)
    time.sleep(1)

