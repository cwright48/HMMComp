# HMMComp

To run, use the script runHMM.py in python dir.

These are the args :

    parser.add_argument("--midiPath", type=str, help="absolute path of the midi file")
    parser.add_argument("--modelPath", type=str, help="absolute path of the model file. Should be .npz"
                                                      "If training, it will be overwritten")
    parser.add_argument("--obsDictPath", type=str, help="absolute path of extra model info file. Should be .npz")
    parser.add_argument("--outputFilePath", type=str, default=None, help="absolute path for generated midi file")
    parser.add_argument("-t", "--train", type=bool, default=True, help="boolean flag if you want to train")
    parser.add_argument("-g", "--generate", type=bool, default=True, help="boolean flag if you want to generate a piece")