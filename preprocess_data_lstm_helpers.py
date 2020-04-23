import json
import os
import shutil

from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import RobustScaler
# TODO: Try different normalizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from operator import itemgetter
from utils import Utils

class PreprocessDataLSTMHelpers():
    '''Class exists for the purpose of preprocessing the data
    suitable for use by pytorch LSTM neural net'''

    @staticmethod
    def preprocess_batched_sequences(batch_size=10, num_songs=1, sequence_length=16):
        """Preprocess the raw data and store them in files as batches"""
        (train_batches, test_batches) = PreprocessDataLSTMHelpers.__get_all_sequences_of_all_songs(num_songs=num_songs, batch_size=batch_size, sequence_length=sequence_length)
        # (train_batches, test_batches) = PreprocessDataLSTMHelpers.__train_test_split(all_sequences_of_all_songs, batch_size=batch_size)
        PreprocessDataLSTMHelpers.__store_train_test_batch(train_batches, test_batches)

    @staticmethod
    def load_preprocessed_data(batch_num, file_label="train"): # train of test
        '''Load an existing batch file after it thas been preprocessed and stored, per prepocess_batched_sequence()'''
        batch_num_as_string = str(batch_num).zfill(8)
        filename = f"./output/preprocessed/{file_label}/batch_{batch_num_as_string}.json"

        if not os.path.exists(filename):
            print(f"{filename} file does not exist")
            return []

        with open(filename) as open_file:
            batch = json.load(open_file)

        return batch

    @staticmethod
    def __preprocess_input_key_signature_mode(key_signature_mode):
        '''The key signature is provided as major/minor. Set to 1 and 0, respectively'''
        if key_signature_mode == "major":
            return 1
        elif key_signature_mode == "minor":
            return -1
        else:
            raise Exception(f"Unknown key signature [{key_signature_mode}]")

    @staticmethod
    def __preprocess_input_is_rest(is_rest):
        '''A rest is an absence of a sound. If rest, set to 1, otherwise 0'''
        if is_rest:
            return 1
        else:
            return -1

    @staticmethod
    def __preprocess_output_note_staff(note_staff):
        '''The staff is the hand used to play the note.
        For right and left hands, set to 1 and 0, respectively'''
        if note_staff == "1":
            return 1
        elif note_staff == "2":
            return -1
        else:
            raise Exception(f"Unknown note_staff [{note_staff}]")

    @staticmethod
    def __define_all_inputs_and_all_outputs(note_index, song_data, note_staff_list):
        '''
        A sequence is a series of notes in order.
        We want to predict the staff for a note (given the notes after it)
        TODO: consider 16 notes before, after, or surrounding it instead (i.e. neighbor notes)
        '''
        if len(song_data) >= (note_index + 16):
            # Provide the note and the 16 following notes
            # to help predict the staff of the target note
            input_sequence = song_data[note_index:note_index+16]
            truth = note_staff_list[note_index:note_index+16]

            return (input_sequence, truth)

        return None

    @staticmethod
    def __generate_note_datapoint(datum):
        '''From source data, generate a datapoint for a note'''
        input_datum = [
            datum["note_duration"],
            PreprocessDataLSTMHelpers.__preprocess_input_is_rest(datum["note_is_rest"]),
            datum["note_midi_pitch"] if ("note_midi_pitch" in datum) and (datum["note_midi_pitch"] is not None) else -1,
            datum["note_time_position"]
        ]

        output_datum = [
            PreprocessDataLSTMHelpers.__preprocess_output_note_staff(datum["note_staff"])
        ]

        return (input_datum, output_datum)

    @staticmethod
    def __load_song_sample(song_index):
        '''From a song, load all the notes from a song, process them and return the list of notes sorted by time'''
        data = []
        song_index_formatted = str(song_index).zfill(4)
        filename = f'./output/music_data/{song_index_formatted}.json'

        if not os.path.exists(filename):
            print(f"{filename} file does not exist")
            return []

        print(f"Processing song#: {song_index}")
        with open(filename) as open_file:
            data = json.load(open_file)

        sorted_data = sorted(data, key=itemgetter("note_time_position"))

        return [
            PreprocessDataLSTMHelpers.__generate_note_datapoint(datum)
            for datum in sorted_data
        ]
    
    @staticmethod
    def __generate_all_viable_sequences_from_song(song_data, sequence_length=16):
        '''Create batch elements of 16 notes'''
        all_viable_sequences = []

        for note_index in range(0, len(song_data)-sequence_length):
            sequence = song_data[note_index:note_index+sequence_length]
            all_viable_sequences.append(sequence)

        return all_viable_sequences

    @staticmethod
    def __get_all_sequences_of_all_songs(num_songs=1, batch_size=1, sequence_length=16):
        '''import all song sequence combinations for all songs'''
        all_songs = []

        # Get all data from all songs from file read
        for song_index in range(0, num_songs):
            song_data = PreprocessDataLSTMHelpers.__load_song_sample(song_index)

            if (not song_data):
                continue
            
            all_songs.append(song_data)
        
        train_sequences_shuffled, test_sequences_shuffled = PreprocessDataLSTMHelpers.__normalize_the_inputs_and_split_train_test(all_songs, sequence_length=sequence_length)

        # Batch the shuffled sequences
        batched_train_sequences_input_output = PreprocessDataLSTMHelpers.batch_sequence(train_sequences_shuffled, batch_size=batch_size)
        batched_test_sequences_input_output = PreprocessDataLSTMHelpers.batch_sequence(test_sequences_shuffled, batch_size=batch_size)
    
        return (batched_train_sequences_input_output, batched_test_sequences_input_output)

    @staticmethod
    def batch_sequence(sequences, batch_size=1):
        # Batch the shuffled sequences
        batched_sequences = PreprocessDataLSTMHelpers.__create_batches_from_all_sequences(sequences, batch_size)

        # each batch is separated by (input, output)
        batched_sequences_input_output = [
            PreprocessDataLSTMHelpers.__separate_input_and_output_from_batch(batch)
            for batch in batched_sequences
        ]

        return batched_sequences_input_output

    @staticmethod
    def __create_batches_from_all_sequences(all_sequences_of_all_songs, batch_size):
        '''Batch the list of sequences [all sequences] into some number of sequences [[batch][batch]]'''
        return Utils.chunks(all_sequences_of_all_songs, batch_size)

    @staticmethod
    def __separate_input_and_output_from_batch(batch):
        '''The batches contain inputs and outputs. Separate them into (batched_inputs, batched_outputs), stratified'''
        batched_input_sequences = [
            [input_features for (input_features, _) in sequence]
                for sequence in batch
        ]

        batched_output_sequences = [
            [output_features for (_, output_features) in sequence]
                for sequence in batch
        ]

        return batched_input_sequences, batched_output_sequences

    @staticmethod
    def __get_standardizer():
        return StandardScaler()

    @staticmethod
    def __get_scaler():
        return MinMaxScaler(feature_range=(-1, 1))

    @staticmethod
    def __get_noop():
        class Noop():
            def partial_fit(self, _datum):
                return self
            def transform(self, datum):
                np_array = array(datum)

                return np_array
        
        return Noop()

    @staticmethod
    def normalize_input_features(input_features, normalizers):
        result = [
            normalizers[feature_index]
                .transform(
                    [[input_features[feature_index]]]
                ).tolist()[0][0]
            for feature_index in range(0, len(input_features))
        ]
        
        return result

    @staticmethod
    def get_all_normalizers():
        return [
            PreprocessDataLSTMHelpers.__get_standardizer(), # 0 note_duration (StandardScaler)
            PreprocessDataLSTMHelpers.__get_noop(), # 1 is_rest (skip)
            # PreprocessDataLSTMHelpers.__get_noop(), # 2 is_note_midi_pitch (skip)
            PreprocessDataLSTMHelpers.__get_scaler(), # note_midi_pitch (MinMaxScaler)
            # PreprocessDataLSTMHelpers.__get_standardizer(), # 4 note_midi_ticks (StandardScaler) -> make None 0
            # PreprocessDataLSTMHelpers.__get_standardizer(), # 5 note_seconds (StandardScaler)
            PreprocessDataLSTMHelpers.__get_scaler() # 6 note_time_position (MinMaxScaler)
        ]
    @staticmethod
    def __normalize_the_inputs_and_split_train_test(all_songs, sequence_length=16):
        train_songs, test_songs = train_test_split(
            all_songs,
            # random_state=0, # make it consistent to start
            # shuffle=True
            shuffle=False
        )

        normalizers = PreprocessDataLSTMHelpers.get_all_normalizers()

        # fit the normalizer to individual data points (consider using pandas to make this more efficient)
        for song_data in train_songs:
            for (input_data, _) in song_data:
                for feature_index, feature_datum in enumerate(input_data):
                    normalizers[feature_index] = normalizers[feature_index].partial_fit([[feature_datum]])
        
        train_songs_scaled = [
            [(PreprocessDataLSTMHelpers.normalize_input_features(input_features, normalizers), output)
                for (input_features, output) in song_data]
                    for song_data in train_songs
        ]

        test_songs_scaled = [
            [(PreprocessDataLSTMHelpers.normalize_input_features(input_features, normalizers), output)
                for (input_features, output) in song_data]
                    for song_data in test_songs
        ]

        train_songs_sequence_combos = [
            shuffle(PreprocessDataLSTMHelpers.__generate_all_viable_sequences_from_song(song_data, sequence_length=sequence_length))[:50]
                for song_data in train_songs_scaled
        ]

        test_songs_sequence_combos = [
            shuffle(PreprocessDataLSTMHelpers.__generate_all_viable_sequences_from_song(song_data, sequence_length=sequence_length))[:50]
                for song_data in test_songs_scaled
        ]

        train_sequences_flattened = [sequence for sequences in train_songs_sequence_combos for sequence in sequences]
        test_sequences_flattened = [sequence for sequences in test_songs_sequence_combos for sequence in sequences]

        # Shuffle training/testing sequences (they are ordered)
        train_sequences_shuffled = shuffle(train_sequences_flattened)
        test_sequences_shuffled = shuffle(test_sequences_flattened)

        return (train_sequences_shuffled, test_sequences_shuffled)

    @staticmethod
    def __train_test_split(all_songs, batch_size=1):
        '''From an array of songs, [[sequences for song 1], [sequences for song 2],...]
        split the data into training and test, batch them, and return (train_batches, test_batches)
        which are both tuples of (input_sequences, output_sequences)'''
        train_songs, test_songs = train_test_split(
            all_songs,
            # random_state=0, # make it consistent to start
            shuffle=True
        )
        train_sequences = [item for sublist in train_songs for item in sublist]
        test_sequences = [item for sublist in test_songs for item in sublist]

        train_sequences_shuffled = shuffle(train_sequences)
        test_sequences_shuffled = shuffle(test_sequences)

        batched_train_sequences = PreprocessDataLSTMHelpers.__create_batches_from_all_sequences(train_sequences_shuffled, batch_size)
        batched_test_sequences = PreprocessDataLSTMHelpers.__create_batches_from_all_sequences(test_sequences_shuffled, batch_size)

        # each batch is separated by (input, output)
        batched_train_sequences_input_output = [
            PreprocessDataLSTMHelpers.__separate_input_and_output_from_batch(batch)
            for batch in batched_train_sequences
        ]

        batched_test_sequences_input_output = [
            PreprocessDataLSTMHelpers.__separate_input_and_output_from_batch(batch)
            for batch in batched_test_sequences
        ]

        return (batched_train_sequences_input_output, batched_test_sequences_input_output)

    @staticmethod
    def __store_train_test_batch(train_batches, test_batches):
        '''Store the data as json for debugging later'''
        source_data = [
            (train_batches, "train"),
            (test_batches, "test")
        ]

        for (batches, file_label) in source_data:
            for batch_num, batch in enumerate(batches):
                print(f"Storing {file_label} batch# {batch_num} data as json file")
                batch_num_as_string = str(batch_num).zfill(8)
                filename_json = Utils.build_dir_path(f"preprocessed/{file_label}/batch_{batch_num_as_string}.json")
                json_data = json.dumps(batch, indent=4)

                with open(filename_json, "w") as outfile:
                    outfile.write(json_data)
