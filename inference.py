import json
import argparse

from command_center import CommandCenter

def inference(model_path, input_midi_file, output_midi_file):
    with open("./midi_hands/example/trains.example.json") as open_file:
        trains = json.load(open_file)

    # Select the correct parameters this model was previously trained in, before inference
    train = trains[0]

    CommandCenter.load_model_for_inference(
        lstm_model_params=train["lstm_model_params"],
        model_layers=train["model_layers"],
        optimization_params=train["optimization_params"],
        model_path=model_path,
        midi_input_filename=input_midi_file,
        midi_output_filename=output_midi_file
    )

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='Add left and right handedness metadata to a Midi track.')
    PARSER.add_argument("model_path", help="Provide a path to the model checkpoint.")
    PARSER.add_argument("input_midi_file", help="Provide a path to your input midi file.")
    PARSER.add_argument("output_midi_file", help="Provide a desired path for your output midi file with handedness metadata.")
    ARGS = PARSER.parse_args()
    inference(ARGS.model_path, ARGS.input_midi_file, ARGS.output_midi_file)
