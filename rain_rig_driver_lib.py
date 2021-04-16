from PCA9685 import PCA9685
from contextlib import contextmanager

# Motor driving direction, [0, 1] is forward, [1, 0] is backward
_Dir = [0, 1]
_max_speed = 100
_min_speed = 30


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

    def RunMotorA(self, speed) -> None:
        validate_speed(speed)
        self.pwm.setDutycycle(self.PWMA, speed)
        self.pwm.setLevel(self.AIN1, _Dir[0])
        self.pwm.setLevel(self.AIN2, _Dir[1])

    def RunMotorB(self, speed) -> None:
        validate_speed(speed)
        self.pwm.setDutycycle(self.PWMB, speed)
        self.pwm.setLevel(self.BIN1, _Dir[0])
        self.pwm.setLevel(self.BIN2, _Dir[1])

    def StopMotors(self):
        self.pwm.setDutycycle(self.PWMA, 0)
        self.pwm.setDutycycle(self.PWMB, 0)


def validate_speed(speed) -> None:
    assert speed <= _max_speed, "speed cannot exceed {} %".format(_max_speed)
    assert speed >= _min_speed, "speed cannot be below {} %".format(_min_speed)


@contextmanager
def run_motor() -> MotorDriver:
    _motor = MotorDriver()
    try:
        yield _motor
    finally:
        _motor.StopMotors()
