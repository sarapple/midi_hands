import os

import torchaudio
from pydub import AudioSegment
from utils import Utils

class PreprocessDataAudioChunkHelpers:
    """Process audio files into n second chunks and save the files"""

    @staticmethod
    def export_chunked_song(song_index, chunk_ms_length):
        """Generate .wav chunk files from a single song, split into millisecond intervals"""
        filename = f"./source_data/wav/00{song_index:02d}.wav"
        if not os.path.exists(filename):
            print(f"{filename} file does not exist")
            return

        print(f"Processing file {filename}")
        chunked_audio_list = PreprocessDataAudioChunkHelpers.__split_audio_into_chunked_list(
            filename,
            chunk_ms_length
        )
        for chunk_index, _ in enumerate(chunked_audio_list):
            PreprocessDataAudioChunkHelpers.__export_audio(
                chunked_audio_list[chunk_index],
                song_index,
                chunk_index
            )

    @staticmethod
    def __load_audio_into_tensor(filename):
        """Load the audio file into torchaudio
        Likely will not need, we can use maestro program to detect notes"""
        waveform, _sample_rate = torchaudio.load(filename)
        tensor = waveform.t()

        return tensor
        # plt.figure()
        # plt.plot(waveform.t().numpy())
        # plt.savefig(f'./output/figures/{filename}.png')

    @staticmethod
    def __split_audio_into_chunked_list(filename, chunk_ms_length):
        """Split audio into pydub chunks, for a given number of milliseconds"""
        song = AudioSegment.from_wav(filename)

        chunked_audio_list = list(Utils.chunks(song, chunk_ms_length))

        return chunked_audio_list

    @staticmethod
    def __export_audio(pydub_audio, song_index, chunk_index):
        """Given a pydub_audio clip, a .wav file is generated with a file
        that includes the provided song_index and the chunk_index"""
        song_index_formatted = f"00{song_index:02d}"
        chunk_index_formatted = f"00{chunk_index:02d}"
        pydub_audio.export(
            Utils.build_dir_path(f"song_chunks/{song_index_formatted}_{chunk_index_formatted}.wav"),
            format="wav"
        )
