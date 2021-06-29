from contextlib import contextmanager
from typing import Optional
import numpy as np
import json
import time

from PCA9685 import PCA9685
from ADS1263 import ADS1263


class MotorDriver:
    def __init__(self):
        #  code taken from the PCA sample code
        self.pwm = PCA9685(0x40, debug=False)
        self.pwm.setPWMFreq(50)
        self.PWMA = 0
        self.AIN1 = 1
        self.AIN2 = 2
        self.PWMB = 5
        self.BIN1 = 3
        self.BIN2 = 4

        # pre-declaring rain rig specific variables to fill from json
        self.filter_constant: Optional[float] = None  # infinite impulse response filter constant between 0 and 1
        self.kp: Optional[float] = None  # PID proportional constant
        self.ki: Optional[float] = None  # PID integral constant
        self.kd: Optional[float] = None  # PID derivative constant
        self.i_range: list = [None, None]  # PID integral output range between 0 and 1
        self.d_range: list = [None, None]  # PID derivative output range between 0 and 1
        self.output_range: list = [None, None]  # motor output range between 0 and 1
        self.control_period_s: Optional[int] = None  # PID controller update period in integer seconds
        self.drive_direction: list = [None, None]  # Motor driving direction, [0, 1] is forward, [1, 0] is backward
        self.adc_vref: Optional[float] = None  # reference voltage for ADC, not sure if calibrated
        self.pres_range_vdc: list = [None, None]  # Autex pressure transducer pressure output voltage range
        self.pres_range_psi: list = [None, None]  # Autex pressure transducer pressure detection range
        self.adc_pressure_channel: Optional[int] = None  # Autex pressure transducer channel on ADC chip

        # load pre-declared variable values from json
        self.load_settings_from_json()

        # initialize ADC chip? taken from sample code
        self.adc = ADS1263()
        assert self.adc.ADS1263_init() != -1, "Failed to initialize ADS1263"

    def load_settings_from_json(self, file_name: str = 'rain_rig_settings.json'):
        with open(file_name) as f:
            motor_settings = json.load(f)
        self.filter_constant = motor_settings['filter_constant']
        self.kp = motor_settings['kp']
        self.ki = motor_settings['ki']
        self.kd = motor_settings['kd']
        self.i_range = motor_settings['i_range']
        self.d_range = motor_settings['d_range']
        self.output_range = motor_settings['output_range']
        self.control_period_s = motor_settings['control_period_s']
        self.drive_direction = motor_settings['drive_direction']
        self.adc_vref = motor_settings['adc_vref']
        self.pres_range_vdc = motor_settings['pres_range_vdc']
        self.pres_range_psi = motor_settings['pres_range_psi']
        self.adc_pressure_channel = motor_settings['adc_pressure_channel']

    def set_motor_A_pressure(self, pres_target: float, duration: int):
        pres_last = self.get_pressure()
        start_time = time.monotonic()

        i_out = 0

        # PID pressure control
        for t in range(0, int(duration), int(self.control_period_s)):
            # IIR filter is applied here when calculating current pressure
            pres_now = self.get_pressure() * self.filter_constant + pres_last * (1 - self.filter_constant)
            pres_diff = pres_target - pres_now

            # calculate PID output
            p_out = self.kp * pres_diff
            i_out = np.clip(i_out + self.ki * pres_diff * self.control_period_s, self.i_range[0], self.i_range[1])
            d_out = np.clip(self.kd * (pres_now - pres_last) / self.control_period_s, self.d_range[0], self.d_range[1])
            pid_out = np.clip(p_out + i_out + d_out, self.output_range[0], self.output_range[1])

            # set motor duty cycle indefinitely until the next update
            self.set_motor_A_duty_cycle(pid_out)
            pres_last = pres_now
            print("Time Elapsed [s]: {}, Pressure [psi]: {:.3f}, Duty cycle: {:.3f}".format(t, pres_now, pid_out))
            time.sleep(t + self.control_period_s - (time.monotonic() - start_time))

    def set_motor_A_duty_cycle(self, duty_cycle: int, duration: Optional[int] = None):
        self.validate_duty_cycle(duty_cycle)  # validate whether duty cycle is within bound

        # set motor duty cycle
        self.pwm.setDutycycle(self.PWMA, duty_cycle)
        self.pwm.setLevel(self.AIN1, self.drive_direction[0])
        self.pwm.setLevel(self.AIN2, self.drive_direction[1])

        # displays filtered pressure every pid period in fixed duty cycle mode
        if duration:
            pres_last = self.get_pressure()
            start_time = time.monotonic()
            for t in range(0, int(duration), int(self.control_period_s)):
                pres_now = self.get_pressure() * self.filter_constant + pres_last * (1 - self.filter_constant)
                pres_last = pres_now
                print("Time Elapsed [s]: {}, Pressure [psi]: {:.3f}, Duty cycle: {:.3f}".format(
                    t, pres_now, duty_cycle))
                time.sleep(t + self.control_period_s - (time.monotonic() - start_time))

    def stop_motors(self):
        self.pwm.setDutycycle(self.PWMA, 0)
        self.pwm.setDutycycle(self.PWMB, 0)

    def validate_duty_cycle(self, duty_cycle):
        assert duty_cycle <= self.output_range[1], "Duty cycle cannot exceed {}".format(self.output_range[1])
        assert duty_cycle >= self.output_range[0], "Duty cycle cannot be below {}".format(self.output_range[0])

    def get_pressure(self) -> float:
        pres_raw = self.adc.ADS1263_GetChannalValue(self.adc_pressure_channel)
        pres_vdc = pres_raw * self.adc_vref / 0x7fffffff
        # quick and dirty vdc -> psi interpolation
        # needs to be written this way to sometimes extrapolate for negative pressure, need improvement
        pres_psi = (pres_vdc - self.pres_range_vdc[0]) / (self.pres_range_vdc[1] - self.pres_range_vdc[0]) * \
                   (self.pres_range_psi[1] - self.pres_range_psi[0]) + self.pres_range_psi[0]
        return pres_psi


@contextmanager
# drive motor through a context manager to avoid motor being left on in case of interruption
def run_motor() -> MotorDriver:
    _motor = MotorDriver()
    try:
        print('Enabling motors')
        yield _motor
    finally:
        print('Disabling motors')
        _motor.stop_motors()
