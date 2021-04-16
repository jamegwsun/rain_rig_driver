from contextlib import contextmanager
from typing import Optional
from numpy import clip
import json
import time

from PCA9685 import PCA9685
from ADS1263 import ADS1263

_Dir = [0, 1]  # Motor driving direction, [0, 1] is forward, [1, 0] is backward
_REF = 5.08  # reference voltage for ADC, uncalibrated?
_PRES_RANGE_PSI = [0, 100]  # Autex pressure transducer pressure detection range
_PRES_RANGE_VDC = [0.5, 4.5]  # Autex pressure transducer pressure output voltage range
_CTRL_PERIOD: int = 1  # PID controller period


class MotorDriver:
    def __init__(self):
        self.pwm = PCA9685(0x40, debug=False)
        self.pwm.setPWMFreq(50)
        self.PWMA = 0
        self.AIN1 = 1
        self.AIN2 = 2
        self.PWMB = 5
        self.BIN1 = 3
        self.BIN2 = 4

        self.filter_constant: Optional[float] = None
        self.kp: Optional[float] = None
        self.ki: Optional[float] = None
        self.kd: Optional[float] = None
        self.i_range: list = [None, None]
        self.d_range: list = [None, None]
        self.output_range: list = [None, None]
        self.load_settings_from_json()

        self.adc = ADS1263()
        assert self.adc.ADS1263_init() != -1, "Failed to initialize ADS1263"

    def load_settings_from_json(self, file_name: str = 'rain_rig_control_settings.json'):
        with open(file_name) as f:
            motor_settings = json.load(f)
        self.filter_constant = motor_settings['filter_constant']
        self.kp = motor_settings['kp']
        self.ki = motor_settings['ki']
        self.kd = motor_settings['kd']
        self.i_range = motor_settings['i_range']
        self.d_range = motor_settings['d_range']
        self.output_range = motor_settings['output_range']

    def set_motor_A_pressure(self, pres_target: float, duration: int):
        start_time = time.monotonic()
        i_out = 0
        pres_last = self.get_pressure()
        # PID
        for t in range(0, int(duration), int(_CTRL_PERIOD)):
            pres_now = self.get_pressure() * self.filter_constant + pres_last * (1 - self.filter_constant)
            pres_diff = pres_target - pres_now
            p_out = self.kp * pres_diff
            i_out = clip(i_out + self.ki * pres_diff * _CTRL_PERIOD, self.i_range[0], self.i_range[1])
            d_out = self.kd * (pres_now - pres_last) / _CTRL_PERIOD
            pid_out = clip(p_out + i_out + d_out, self.output_range[0], self.output_range[1])
            self.set_motor_A_duty_cycle(pid_out)
            pres_last = self.get_pressure()
            print(pres_now, pid_out, time.monotonic(), p_out, i_out, d_out)
            time.sleep(t + _CTRL_PERIOD - (time.monotonic() - start_time))

    def set_motor_A_duty_cycle(self, duty_cycle: int):
        self.validate_duty_cycle(duty_cycle)
        self.pwm.setDutycycle(self.PWMA, duty_cycle)
        self.pwm.setLevel(self.AIN1, _Dir[0])
        self.pwm.setLevel(self.AIN2, _Dir[1])

    def set_motor_B_duty_cycle(self, duty_cycle: int):
        self.validate_duty_cycle(duty_cycle)
        self.pwm.setDutycycle(self.PWMB, duty_cycle)
        self.pwm.setLevel(self.BIN1, _Dir[0])
        self.pwm.setLevel(self.BIN2, _Dir[1])

    def stop_motors(self):
        self.pwm.setDutycycle(self.PWMA, 0)
        self.pwm.setDutycycle(self.PWMB, 0)

    def validate_duty_cycle(self, duty_cycle):
        assert duty_cycle <= self.output_range[1], "Duty cycle cannot exceed {}".format(self.output_range[1])
        assert duty_cycle >= self.output_range[0], "Duty cycle cannot be below {}".format(self.output_range[0])

    def get_pressure(self) -> float:
        pres_raw = self.adc.ADS1263_GetChannalValue(0)
        pres_vdc = pres_raw * _REF / 0x7fffffff
        pres_psi = (pres_vdc - _PRES_RANGE_VDC[0]) / (_PRES_RANGE_VDC[1] - _PRES_RANGE_VDC[0]) * \
                   (_PRES_RANGE_PSI[1] - _PRES_RANGE_PSI[0]) + _PRES_RANGE_PSI[0]
        return pres_psi


@contextmanager
def run_motor() -> MotorDriver:
    _motor = MotorDriver()
    try:
        print('Enabling motors')
        yield _motor
    finally:
        print('Disabling motors')
        _motor.stop_motors()
