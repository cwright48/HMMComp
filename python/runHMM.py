import argparse

from hmm import HMM
import songData as sData


def train(midiFilePath, modelFilePath, obsDictFilePath):
    song = sData.readMidi(midiFilePath)
    newFormat = sData.convertMidiToSongData(song)
    hmm = HMM(len(newFormat.tracks), 50, 128 * 2 * 24)
    hmm.initialize()

    hmm.train(newFormat, maxEpochs=500, filePath=modelFilePath,
              obsDictFilePath=obsDictFilePath)

def generate(midiFilePath, modelFilePath, obsDictFilePath, outputFilePath, length=100):
    song = sData.readMidi(midiFilePath)
    newFormat = sData.convertMidiToSongData(song)
    hmm = HMM(len(newFormat.tracks), 50, 2 * 128 * 24, modelFilePath, obsDictFilePath=obsDictFilePath)
    newSong = hmm.generateSequence(length)
    newFormat.setTracks(newSong)

    sData.convertSongDataToMidi(newFormat, outputFilePath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--midiPath", type=str, help="absolute path of the midi file")
    parser.add_argument("--modelPath", type=str, help="absolute path of the model file. Should be .npz"
                                                      "If training, it will be overwritten")
    parser.add_argument("--obsDictPath", type=str, help="absolute path of extra model info file. Should be .npz")
    parser.add_argument("--outputFilePath", type=str, default=None, help="absolute path for generated midi file")
    parser.add_argument("-t", "--train", type=bool, default=True, help="boolean flag if you want to train")
    parser.add_argument("-g", "--generate", type=bool, default=True, help="boolean flag if you want to generate a piece")
    args = parser.parse_args()

    if args.train:
        train(args.midiPath, args.modelPath, args.obsDictPath)

    if args.generate:
        generate(args.midiPath, args.modelPath, args.obsDictPath, args.outputFilePath)
