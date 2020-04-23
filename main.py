import json

from preprocess_data_lstm_helpers import PreprocessDataLSTMHelpers
from audio_model_builder import AudioModelBuilder
from command_center import CommandCenter

def main():
    '''Run the codeeee mannn'''
    # params = AudioModelBuilder.get_hyper_parameters()
    # default_lstm_model_params = params["lstm_model_params"]
    # default_optimization_params = params["optimization_params"]
    # default_model_layers = AudioModelBuilder.get_model_layers()
    # default_model_name = "baseline"
    # default_batch_size = default_lstm_model_params["batch_size"]

    num_songs = 288

    # Import musicxml files as json
    # CommandCenter.ingest_data(num_songs)
    
    # Take json files and batch sequences with note datapoints we care about
    # PreprocessDataLSTMHelpers.preprocess_batched_sequences(batch_size=100, num_songs=num_songs, sequence_length=16)

    # Chunk song into smaller chunks (may not use)
    # PreprocessDataAudioChunkHelpers.export_chunked_song(1, 10000)

    # Open a training config and get all trains
    with open("./trains.json") as open_file:
        trains = json.load(open_file)

    
    # Train each model for each training config
    for train in trains:
        # CommandCenter.start_model_training(
        #     **train
        # )

        CommandCenter.load_model_for_inference(
            model_name=train["model_name"],
            lstm_model_params=train["lstm_model_params"],
            model_layers=train["model_layers"],
            optimization_params=train["optimization_params"],
            epochs=6,
            midi_input_filename="source_data/simple_midi/ready.mid",
            midi_output_filename="left_right_handedness_fixed.mid"
        )

        # CommandCenter.load_model_to_confirm_evaluation(
        #     model_name=train["model_name"],
        #     lstm_model_params=train["lstm_model_params"],
        #     model_layers=train["model_layers"],
        #     optimization_params=train["optimization_params"]
        # )
    
        # CommandCenter.load_model_and_save_to_mobile(
        #     model_name=train["model_name"],
        #     epochs=7,
        #     lstm_model_params=train["lstm_model_params"],
        #     model_layers=train["model_layers"],
        #     optimization_params=train["optimization_params"]
        # )


    # When things aren't working, try running this.
    # This trains on the first two batches only,
    # and then tests against the training batch (it should match)
    # for train in trains:
    #     CommandCenter.sanity_check_start_model_training(
    #         **train
    #     )
        # CommandCenter.sanity_check_load_model_to_confirm_evaluation(
        #     model_name=train["model_name"],
        #     file_label="train",
        #     lstm_model_params=train["lstm_model_params"],
        #     model_layers=train["model_layers"],
        #     optimization_params=train["optimization_params"]
        # )
    # for train in trains:
    #     CommandCenter.load_model_and_save_to_mobile(
    #         model_name=train["model_name"],
    #         epochs=468,
    #         lstm_model_params=train["lstm_model_params"],
    #         model_layers=train["model_layers"],
    #         optimization_params=train["optimization_params"]
    #     )


if __name__ == "__main__":
    main()
