#!/usr/bin/env python3
"""
Class: Intelligent Robotics
Author: David Hawbaker
"""
from adafruit_servokit import ServoKit


class ServoInterface:
    def __init__(self):
        self.kit = ServoKit(channels=8)

    def interpret_classification(self, classification):
        """ Interpret NN Classification

        :param classification: class to interpret
        :return: None
        """
        if classification == 0:
            pass
        elif classification == 1:
            self.make_fist()
        elif classification == 2:
            self.open_hand()
        elif classification == 3:
            self.rock_on()

    def set_thumb(self, angle):
        """ Set the angle of the thumb 30 is closed 110 is open

        :param angle: 30-110
        :return: None
        """
        if angle < 30:
            angle = 30
        elif angle > 110:
            angle = 110
        self.kit.servo[0].angle = angle

    def set_index(self, angle):
        """ Set the angle of the index 30 is closed 110 is open

        :param angle: 30-110
        :return: None
        """
        if angle < 30:
            angle = 30
        elif angle > 110:
            angle = 110
        self.kit.servo[1].angle = angle

    def set_middle(self, angle):
        """ Set the angle of the middle 30 is closed 110 is open

        :param angle: 30-110
        :return: None
        """
        if angle < 30:
            angle = 30
        elif angle > 110:
            angle = 110
        self.kit.servo[5].angle = angle

    def set_ring(self, angle):
        """ Set the angle of the middle 30 is closed 110 is open

        :param angle: 30-110
        :return: None
        """
        if angle < 30:
            angle = 30
        elif angle > 110:
            angle = 110
        self.kit.servo[4].angle = angle

    def set_pinky(self, angle):
        """ Set the angle of the middle 30 is closed 110 is open

        :param angle: 30-110
        :return: None
        """
        if angle < 30:
            angle = 30
        elif angle > 110:
            angle = 110
        self.kit.servo[2].angle = angle

    def set_wrist(self, angle):
        """ Set the angle of the wrist 0 is left 180 is right

        :param angle: 0-180
        :return: None
        """
        if angle < 0:
            angle = 0
        elif angle > 180:
            angle = 180
        self.kit.servo[6].angle = angle

    def make_fist(self):
        """ Close all fingers to make a fist

        :return: None
        """
        self.set_index(30)
        self.set_middle(30)
        self.set_ring(30)
        self.set_pinky(30)
        self.set_thumb(30)

    def open_hand(self):
        """ Open all of the fingers

        :return: None
        """
        self.set_thumb(110)
        self.set_index(110)
        self.set_middle(110)
        self.set_ring(110)
        self.set_pinky(110)

    def make_peace(self):
        """ Make the peace sign

        :return: None
        """
        self.set_index(110)
        self.set_middle(110)
        self.set_ring(30)
        self.set_pinky(30)
        self.set_thumb(30)

    def rock_on(self):
        """ Rock on

        :return: None
        """
        self.set_index(110)
        self.set_middle(30)
        self.set_ring(30)
        self.set_pinky(110)
        self.set_thumb(30)
