from mido import MidiFile

def readMidi(filePath):
    """
    Reads in a MIDI file.

    :param filePath: path to file
    :return: MidiFile object
    """

    # TODO: error checking on path, etc.
    file = MidiFile(filePath)
    return file