import os
import json

from midi_hands.libs.magenta_musicxml.magenta_musicxml_parser import MusicXMLDocument
from midi_hands.utils import Utils

class PreprocessDataMusicXMLHelpers():
    '''Take music XML files and convert them to json data files'''
    @staticmethod
    def process_musicxml(song_index):
        """Generate json files from musicxml files"""
        song_index_formatted = str(song_index).zfill(4)
        filename = f"./source_data/musicxml/{song_index_formatted}.musicxml"
        if not os.path.exists(filename):
            print(f"{filename} file does not exist")
            return
        print(f"Processing {filename}")
        musicxml_obj: MusicXMLDocument = PreprocessDataMusicXMLHelpers.__parse_musicxml(filename)
        datum_list = PreprocessDataMusicXMLHelpers.__generate_music_dict(musicxml_obj)
        json_data = json.dumps(datum_list, sort_keys=True, indent=4)

        filename_json = Utils.build_dir_path(f"music_data/{song_index_formatted}.json")
        with open(filename_json, "w") as outfile:
            outfile.write(json_data)

    @staticmethod
    def __parse_musicxml(filename):
        """Parse the file into a music xml object"""
        xml_document = MusicXMLDocument(filename)

        return xml_document

    @staticmethod
    def __generate_music_dict(magenta_musicxml_instance):
        """Generate a data dictionary for the given musicxml with relevant datapoints"""
        datum_list = []

        for part in magenta_musicxml_instance.parts:
            for measure in part.measures:
                # Implement precise logic using
                # musicxml_obj.get_time_signatures() and musicxml_obj.get_key_signatures
                if measure.time_signature:
                    last_known_time_signature = measure.time_signature
                if measure.key_signature:
                    last_known_key_signature = measure.key_signature

                for note in measure.notes:
                    datum = {}
                    # datum["song_total_time_secs"] = magenta_musicxml_instance.total_time_secs
                    # datum["song_index"] = song_index
                    datum["measure_time_signature_numerator"] = last_known_time_signature.numerator
                    datum["measure_time_signature_denominator"] = last_known_time_signature.denominator
                    datum["measure_key_signature_mode"] = last_known_key_signature.mode
                    datum["measure_key_signature_key"] = last_known_key_signature.key
                    datum["measure_start_time_position"] = measure.start_time_position

                    datum["note_duration"] = note.note_duration.duration
                    datum["note_midi_ticks"] = note.note_duration.midi_ticks
                    datum["note_seconds"] = note.note_duration.seconds

                    if hasattr(note, "staff"):
                        datum["note_staff"] = note.staff
                    if note.is_rest:
                        datum["note_is_rest"] = note.is_rest
                        datum["note_pitch"] = None
                        datum["note_midi_pitch"] = None
                    else:
                        datum["note_is_rest"] = False
                        datum["note_pitch"] = note.pitch[0]
                        datum["note_midi_pitch"] = note.pitch[1]

                    datum["note_voice"] = note.voice
                    datum["note_velocity"] = note.velocity
                    datum["note_time_position"] = note.note_duration.time_position

                    datum_list.append(datum)

        return datum_list
