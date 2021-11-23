#!/usr/bin/python3

from BoardInterface import BoardInterface

bi = BoardInterface()

base_file = '~/projects/psu_fall_2021/intelligent_robotics/brain_interface/data/'
input_file = base_file + 'double_blinks/first_Run/raw_data_1.csv'
output_file = base_file + 'double_blinks/first_Run/raw_data_1_denoise.csv'

data = bi.read_file(input_file)
denoised_data = bi.denoise(data, output_file)
bi.write_data(denoised_data, output_file)
