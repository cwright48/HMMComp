import numpy as np
from mido import MidiFile, MidiTrack, Message, second2tick, tick2second
from dataUtils import readMidi

class SongTimeSeries:

    def __init__(self, fileName, bach=False):

        mFile = readMidi(fileName)

        self.midiMetaTrack = mFile.tracks[0]
        self.midiTracks = mFile.tracks[1:]
        self.ticksPerBeat = mFile.ticks_per_beat
        self.timeInterval = int(self.ticksPerBeat/8)
        self.tempi = {}
        self.tempi[0] = 50000
        self.bach = bach

        for msg in self.midiMetaTrack:
            if msg.type == 'set_tempo':
                self.tempi[int(msg.time/self.timeInterval)] = msg.tempo
                print(msg)

        trackDurationS = 0 #track duration in seconds
        trackDurationT = 0 # track duration in ticks
        tempoTimes = list(self.tempi.keys())
        for tempoTime in range(len(self.tempi.keys()) - 1):
            trackDurationS += tick2second(tempoTime*self.timeInterval, self.ticksPerBeat,
                                          self.tempi[tempoTimes[tempoTime]])
            trackDurationT += tempoTime*self.timeInterval

        lastTempoKey = tempoTimes[-1]

        trackDurationT += second2tick(mFile.length - trackDurationS, self.ticksPerBeat,
                                         self.tempi[lastTempoKey])

        self.trackDuration = int(trackDurationT/self.timeInterval)


    def getTracksAsTimeSeries(self):
        trackNotesInTime = np.zeros((self.trackDuration, len(self.midiTracks)), dtype=int)

        for i, track in enumerate(self.midiTracks):
            offset = 0  # time offset for meta messages
            prev_msg = None
            curTime = 0

            for msg in track:
                if msg.is_meta:
                    continue

                if msg.type != 'note_on' and msg.type != 'note_off':
                    offset += msg.time
                    continue

                if prev_msg is not None:
                    startTime = curTime + int(prev_msg.time/self.timeInterval)
                    if offset > 0:
                        newTime = startTime + int(offset/self.timeInterval) + int(msg.time/self.timeInterval)
                        offset = 0
                    else:
                        newTime = startTime + int(msg.time/self.timeInterval)

                    trackNotesInTime[startTime:newTime, i] = msg.note
                    curTime = newTime
                    prev_msg = None
                    continue

                if msg.velocity > 0 and msg.type == "note_on":
                    prev_msg = msg
                    continue

        return trackNotesInTime

    def setTracksFromTrackNotesInTime(self, trackNotesInTime):

        self.trackDuration = trackNotesInTime.shape[0]
        newTracks = []

        for track in range(trackNotesInTime.shape[1]):
            newTrack = MidiTrack()

            currentNote = 0
            timeSinceChange = 0

            for note in trackNotesInTime[:, track]:
                if note == currentNote:
                    timeSinceChange += 1
                elif currentNote == 0 and note > 0:
                    if not self.bach:
                        timeSinceChange -= 1
                    newTrack.append(Message(type='note_on', channel=track, note=note, velocity=100,
                                            time=(timeSinceChange+1)*self.timeInterval))
                    timeSinceChange = 0
                    currentNote = note
                elif note != currentNote and currentNote != 0:
                    if not self.bach:
                        timeSinceChange -= 1
                    newTrack.append(Message(type='note_off', channel=track, note=note, velocity=0,
                                            time=(timeSinceChange+1)*self.timeInterval))
                    timeSinceChange = 0
                    currentNote = 0
                    if note != 0:
                        if not self.bach:
                            timeSinceChange += 1
                        newTrack.append(Message(type='note_on', channel=track, note=note, velocity=100,
                                                time=timeSinceChange*self.timeInterval))
                        currentNote = note

            newTracks.append(newTrack)

        self.midiTracks = newTracks

    def writeMidiFromTracks(self, fileName):

        mFile = MidiFile()
        mFile.ticks_per_beat = self.ticksPerBeat
        mFile.tracks.append(self.midiMetaTrack)

        for track in self.midiTracks:
            mFile.tracks.append(track)

        mFile.save(fileName)


