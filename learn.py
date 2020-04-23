import json

from preprocess_data_lstm_helpers import PreprocessDataLSTMHelpers
from command_center import CommandCenter

def learn():
    ### PREPOCESS ###
    num_songs = 288

    # Import musicxml files as json
    CommandCenter.ingest_data(num_songs)
    
    # Take json files and batch sequences with note datapoints we care about
    PreprocessDataLSTMHelpers.preprocess_batched_sequences(batch_size=100, num_songs=num_songs, sequence_length=16)

    ### TRAIN ###
    # Open a training config and get all trains
    with open("./trains.json") as open_file:
        trains = json.load(open_file)
    
    # Train each model for each training config
    for train in trains:
        CommandCenter.start_model_training(
            **train
        )

if __name__ == "__main__":
    learn()
