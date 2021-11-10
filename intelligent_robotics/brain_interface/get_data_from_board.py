import argparse
import time
import numpy as np

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams
from brainflow.exit_codes import *
from BoardInterface import BoardInterface

from EEGModels import EEGNet


def main():

    board_i = BoardInterface()
    board_i.start_stream()

    board_i.end_stream()

    while True:
        try:




        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
