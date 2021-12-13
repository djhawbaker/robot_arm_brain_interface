#!/usr/bin/python3
"""
Class: Intelligent Robotics
Author: David Hawbaker

Plot data
"""
import pandas as pd
import plotly.express as px


def main():
    base = '~/projects/psu_fall_2021/intelligent_robotics/brain_interface/data/'
    inputfile = base + 'double_blinks/first_Run/400/split_data_1_aa.csv'
    outputfile = base + 'double_blinks/first_Run/400/split_data_1_aa.png'

    df = pd.read_csv(inputfile, delimiter='\t')
    print(df.columns)
    print(df.head())
    x = df['Time']
    y = df['B']
    fig = px.line(df, y='B', range_x=(0, 400), range_y=(-1300, 0), title='Brain waves')
    fig.show()
    f = open(outputfile, 'w')
    fig.write_image(file=outputfile, format='png')
    f.close()


if __name__ == "__main__":
    main()
