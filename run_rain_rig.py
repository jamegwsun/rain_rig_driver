import time
import argparse

from rain_rig_driver_lib import run_motor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", dest="pressure_psi", type=float, required=True,
        help="Rain rig target pressure, override by duty cycle")
    parser.add_argument(
        "-t", dest="duration_s", type=int, required=True,
        help="Rain rig operation duration in seconds.")
    parser.add_argument(
        "-d", dest="duty_cycle", type=float, required=False, default=0,
        help="Option to directly set duty cycle. Bypasses pressure target and PID control")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with run_motor() as m:
        if args.duty_cycle:
            m.set_motor_A_duty_cycle(args.duty_cycle, args.duration_s)
        else:
            m.set_motor_A_pressure(args.pressure_psi, args.duration_s)
