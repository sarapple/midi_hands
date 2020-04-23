# MIDI Hands

MIDI Left and Right-Handedness Transcription (For Piano)

## What This Does

Takes a midi file and returns a midi file with left and right handedness metadata added to the file.

## How MIDI Works and Goal

Typically, separate tracks ae used to separate left and right hands in a MIDI-version of a song. 

This aims to improve existing tracks without any separation of handedness, as well as generating handedness in MIDI files with a single track.

## How It Learned

A Pytorch LSTM neural net was used to predict left and right handedness on sequences of notes.  Ground truth sheet music in musicxml form (containing left and right handedness) was used to learn. The output predicts either left or right on each note.

## Checkpoint

A checkpoint is made available for others to use in the `/example` folder. Parameters for learning are not hard-coded, and can be adjusted by adding a `trains.json` json file to the root of the project. An example file `/example/trains.example.json` is provided.

## To run the model

- Rename `/example/trains.example.json` to `trains.json` and move it to the root of the folder. C
- Then call the following command:
```
python learn.py
```

## To run the model

- Rename `/example/trains.example.json` to `trains.json` and move it to the root of the folder. C
- Then call the following command:
```
python learn.py
```

## To generate a midi file from an original with left and right hand encodings

- Call the following command:
```
python inference.py my_midi_input.mid my_midi_output.mid
```

## Credits

A helper file from Magenta (Copyright 2020 The Magenta Authors)was pulled out and used, unaltered from the source.  This was done in favor of importing the large lib involved with installing Magenta. This was used for parseing MusicXML into a dictionary.