from pydub import AudioSegment
from pydub.utils import mediainfo

def audio_to_array(filepath):
    audio = AudioSegment.from_file(filepath)

    return audio.get_array_of_samples()

