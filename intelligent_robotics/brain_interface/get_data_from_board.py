import argparse
import time
import numpy as np

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations


def main():
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    params.ip_port = 0
    params.serial_port = '/dev/ttyACM0'
    params.mac_address = 'c0:b9:23:13:2c:be'
    params.other_info = ''
    params.serial_number = ''
    params.ip_address = ''
    params.ip_protocol = 0
    params.timeout = 0
    params.file = 'output.csv'

    board_id = 1  # GANGLION_BOARD

    board = BoardShim(board_id, params)
    board.prepare_session()

    board.start_stream()  # use this for default options
    #board.start_stream(45000, args.streamer_params)
    print("################################################################################")
    print("###                               Streaming                                  ###")
    print("###                          Press Ctrl+c to stop                            ###")
    print("################################################################################")

    while True:
        try:
            # Get data once per second
            time.sleep(1)
            # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
            data = board.get_board_data(num_samples=200)  # get 200 data samples and remove it from internal buffer
            eeg = board.get_eeg_channels(board_id)
            print("eeg channels ", eeg)
            print("size ", data.size)
            print("shape", data.shape)
            DataFilter.write_file(data, params.file, 'w')
            print(data)

        except KeyboardInterrupt:
            break

    board.stop_stream()
    board.release_session()
    print("\n################################################################################")
    print("###                          Successfully Stopped                            ###")
    print("################################################################################")


if __name__ == "__main__":
    main()
