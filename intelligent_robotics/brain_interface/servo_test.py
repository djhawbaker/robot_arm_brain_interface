#!/usr/bin/python3
"""
Class: Intelligent Robotics
Author: David Hawbaker

Use this file to test the robot hand movements
"""
from ServoInterface import ServoInterface


def main():
    print('hi')
    si = ServoInterface()

    while True:
        i = input("select a hand motion: ")
        if i == 'o':
            si.open_hand()
        elif i == 'c':
            si.make_fist()
        elif i == 'p':
            si.make_peace()
        elif i == 'r':
            si.rock_on()
        elif i == 'wl':
            si.set_wrist(0)
        elif i == 'wr':
            si.set_wrist(180)
        else:
            si.open_hand()


if __name__ == "__main__":
    main()
