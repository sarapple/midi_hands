import torch
import json
import mido
from mido import MidiTrack, MidiFile, Message, MetaMessage
from midi_hands.preprocess_data_musicxml_helpers import PreprocessDataMusicXMLHelpers
from midi_hands.preprocess_data_lstm_helpers import PreprocessDataLSTMHelpers
from midi_hands.audio_model_runner import AudioModelRunner
from midi_hands.audio_model_tester import AudioModelTester
from midi_hands.audio_model_builder import AudioModelBuilder
from midi_hands.mobile import Mobile
from midi_hands.utils import Utils
from midi_hands.preprocess_midi_inference_data import PreprocessMIDIInferenceData

class CommandCenter():
    @staticmethod
    def sanity_check_start_model_training(epochs_to_train=200, model_name="base", lstm_model_params=None, optimization_params=None, model_layers=None):
        """SANITY CHECK: Build the model but with two batches"""
        AudioModelTester.sanity_check_build_model(
            model_name=model_name,
            epochs_to_train=epochs_to_train,
            get_batch_data=AudioModelRunner.get_batch_data,
            get_batch_data_length=AudioModelRunner.get_batch_data_length,
            lstm_model_params=lstm_model_params,
            optimization_params=optimization_params,
            model_layers=model_layers
        )

    @staticmethod
    def sanity_check_load_model_to_confirm_evaluation(
        model_name="sanity",
        file_label="test",
        model_layers=None,
        lstm_model_params=None,
        optimization_params=None
    ):
        """SANITY CHECK: Load the built model but with two batches"""
        AudioModelTester.sanity_check_test_model(
            model_name=model_name,
            get_batch_data=AudioModelRunner.get_batch_data,
            file_label=file_label,
            model_layers=model_layers,
            lstm_model_params=lstm_model_params,
            optimization_params=optimization_params
        )

    @staticmethod
    def start_model_training(epochs_to_train=200, model_name="base", lstm_model_params=None, optimization_params=None, model_layers=None):
        """build the model"""
        AudioModelRunner.create_and_train_model(
            model_name=model_name,
            epochs_to_train=epochs_to_train,
            get_batch_data=AudioModelRunner.get_batch_data,
            get_batch_data_length=AudioModelRunner.get_batch_data_length,
            lstm_model_params=lstm_model_params,
            optimization_params=optimization_params,
            model_layers=model_layers
        )
        
    @staticmethod
    def load_model_to_confirm_evaluation(
            model_name="base",
            model_layers=None,
            lstm_model_params=None,
            optimization_params=None
        ):
        '''Load a model that was built previously, and run the test against it'''
        avg_error, avg_accuracy = AudioModelRunner.load_and_test_model(
            model_name=model_name,
            get_batch_data=AudioModelRunner.get_batch_data,
            get_batch_data_length=AudioModelRunner.get_batch_data_length,
            model_layers=model_layers,
            lstm_model_params=lstm_model_params,
            optimization_params=optimization_params
        )

        print(f"Average Error: [{avg_error}]")
        print(f"Average Accuracy: [{avg_accuracy}]")

    @staticmethod
    def resume_model_training(
            epochs_to_train=1,
            model_name="base",
            lstm_model_params=None,
            optimization_params=None,
            model_layers=None
        ):
        """resume building the model"""
        AudioModelRunner.resume_model_train(
            model_name=model_name,
            epochs_to_train=epochs_to_train,
            get_batch_data=AudioModelRunner.get_batch_data,
            get_batch_data_length=AudioModelRunner.get_batch_data_length,
            optimization_params=optimization_params,
            model_layers=model_layers,
            lstm_model_params=lstm_model_params,
        )

    @staticmethod
    def ingest_data(num_songs):
        '''Parse music files and chunk them, and parse songs from musicxml into usable data format'''
        # To chunk the songs into 10s intervals
        # ten_seconds = 10 * 1000
        # for index in range(0, num_songs):
        #     PreprocessDataAudioChunkHelpers.export_chunked_song(index, ten_seconds)

        # To recreate json from songs
        for song_index in range(0, num_songs):
            PreprocessDataMusicXMLHelpers.process_musicxml(song_index)


    @staticmethod
    def load_model_for_inference(
        model_layers=None,
        lstm_model_params=None,
        optimization_params=None,
        midi_input_filename=None,
        midi_output_filename=None,
        model_path=None,
    ):
        '''Load a model that was built previously, and run the test against it'''
        model, _, _, _ = AudioModelBuilder.assemble_existing_lstm_model(
            model_layers=model_layers,
            lstm_model_params=lstm_model_params,
            optimization_params=optimization_params,
            model_path=model_path
        )
        model.eval()
        midifile = PreprocessMIDIInferenceData.get_left_right_hands_midifile(
            model,
            midi_input_filename=midi_input_filename
        )
        midifile.save(file=midi_output_filename)
