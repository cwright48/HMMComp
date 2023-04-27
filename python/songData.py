import numpy as np
from mido import MidiFile, MidiTrack, Message


class SongData:

    def __init__(self, midiMetaTrack, ticksPerBeat, tracks, durations, trackMetaInfo=None):
        self.midiMetaTrack = midiMetaTrack
        self.tracks = tracks
        self.ticksPerBeat = ticksPerBeat
        self.durations = dict((v, k) for k, v in durations.items())
        self.trackMeta = trackMetaInfo

    def getMetaTrack(self):
        return self.midiMetaTrack

    def setTracks(self, tracks):
        self.tracks = tracks

    def getAllPossibleObservations(self):
        observations = {}
        index = 0
        for note in range(0,128):
            for duration in range(len(self.durations)):
                observations[(note, 1, duration, 0, 0)] = index
                observations[(note, 0, duration, 0, 0)] = index+1
                index += 2

        return observations

    def getShortenedTracks(self, cutoff=999999999):
        shortestTrack = cutoff

        for track in self.tracks:
            if len(track) < shortestTrack:
                shortestTrack = len(track)

        newTracks = np.empty((shortestTrack, len(self.tracks)), dtype=tuple)
        for i, track in enumerate(self.tracks):
            for t in range(shortestTrack):
                if t < len(track):
                    newTracks[t, i] = tuple(track[t])
                else:
                    newTracks[t, i] = (0, 0, 0, 0, 0)

        return newTracks

    def getPaddedTracks(self):
        longestTrack = 0

        for track in self.tracks:
            if len(track) > longestTrack:
                longestTrack = len(track)

        paddedTracks = np.empty((longestTrack, len(self.tracks)), dtype=tuple)

        for i, track in enumerate(self.tracks):
            for t in range(longestTrack):
                if t < len(track):
                    paddedTracks[t, i] = tuple(track[t])
                else:
                    paddedTracks[t, i] = (0, 0, 0, 0, 0)

        return paddedTracks

    def getTracks(self):

        return self.tracks


def convertMidiToSongData(midiData):
    """
    Takes data in MIDI format and converts it into a SongData object

    :param midiData:
    :return: SongData
    """
    tracks = []
    beat = midiData.ticks_per_beat
    durations = {int(beat/8):                 0,    # 32nd notes
                 int(beat/4):                 1,    # 16th notes
                 int(beat/4 + beat/8):        2,    # 16th notes
                 int(beat/2):                 3,    # 8th note
                 int(beat/2 + beat/4):        4,    # 8th + 16th note
                 int(beat):                   5,    # quarter notes
                 int(beat + beat/4):          6,    # quarter note + 16th
                 int(beat + beat/2):          7,    # quarter + 8th
                 int(beat + beat/2 + beat/4): 8,    # quarter + 8th + 16th
                 int(2*beat):                 9,    # half notes
                 int(2*beat + beat/4):        10,   # half note + 16th note
                 int(2*beat + beat/2):        11,   # half note + 8th note
                 int(3*beat):                 12,   # half + quarter note
                 int(3*beat + beat/4):        13,   # half + quarter + 16th
                 int(3*beat + beat/2):        14,   # half + quarter + 8th note
                 int(4*beat):                 15,   # whole note
                 int(5*beat):                 16,   # whole + quarter
                 int(6*beat):                 17,   # whole + half
                 int(7*beat):                 18,   # whole + half + quarter
                 int(8*beat):                 19,   # two whole
                 int(9*beat):                 20,   # two whole + half
                 int(10*beat):                21,   # two whole + half + quarter
                 int(11*beat):                22,   # two whole
                 int(12*beat):                23}   # two whole + quarter

    trackMetaData = []
    for track in midiData.tracks[1:]:
        noteSequence = []
        prev_msg = None
        offset = 0
        metaInfo = []

        for msg in track:
            if msg.is_meta:
                continue

            if msg.type != 'note_on' and msg.type != 'note_off':
                print(msg)
                offset += msg.time
                metaInfo.append(msg)
                continue

            if prev_msg is not None:
                vel = 1 if prev_msg.velocity > 0 else 0
                if msg.time in durations:
                    noteSequence.append((msg.note, vel, durations[msg.time], 0, 0))
                elif msg.time + 1 in durations:
                    noteSequence.append((msg.note, vel, durations[msg.time + 1], 0, 0))
                elif msg.time + offset in durations:
                    noteSequence.append((msg.note, vel, durations[msg.time + offset], 0, 0))
                else:
                    print("couldn't do it")
                    noteSequence.append([msg.note, vel, durations[msg.time + prev_msg.time], 0, 0])
                prev_msg = None
                continue

            if msg.velocity > 0 and msg.type == "note_on":
                if offset > 0:
                    msg.time += offset
                    offset = 0
                prev_msg = msg
                continue

        if len(noteSequence) > 0:
            tracks.append(noteSequence)
            trackMetaData.append(metaInfo)

    return SongData(midiData.tracks[0], midiData.ticks_per_beat, np.asarray(tracks), durations, trackMetaData)


def convertSongDataToMidi(songData, saveToFile=None):
    """
    Converts the given SongData to MidiFile

    :param songData:
    :return: MidiFile
    """

    mid = MidiFile()
    mid.ticks_per_beat = songData.ticksPerBeat
    mid.tracks.append(songData.getMetaTrack())
    durations = songData.durations

    for i, track in enumerate(songData.getTracks()):
        newTrack = MidiTrack()

        if i in songData.trackMeta:
            for meta in songData.trackMeta[i]:
                newTrack.append(meta)

        for msg in track:
            vel = 100 if msg[1] > 0 else 0
            newTrack.append(Message(type='note_on', channel=i, note=msg[0], velocity=vel, time=msg[3]))
            if msg[4] > 0:
                newTrack.append(Message(type='note_off', channel=i, note=msg[0], velocity=0, time=durations[msg[2]]-msg[4]))
            else:
                newTrack.append(Message(type='note_off', channel=i, note=msg[0], velocity=0, time=durations[msg[2]]))

        mid.tracks.append(newTrack)

    if saveToFile is not None:
        mid.save(saveToFile)

    return mid

def readMidi(filePath):
    """
    Reads in a MIDI file.

    :param filePath: path to file
    :return: MidiFile object
    """

    # TODO: error checking on path, etc.
    file = MidiFile(filePath)
    return file
