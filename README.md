# HMMComp

To run, use the script runHMM.py in python dir. Running this file with the -t argument will train a new HMM model. Once you have a trained model, you can run it again with -g to generate a song.

These are the args:

    "--midiPath": absolute path of the midi file to use during training.
    "--modelPath": absolute path of the model file. Should be .npz, If training, it will be overwritten.
    "--obsDictPath": absolute path of extra model info file. Should be .npz
    "--outputFilePath: absolute path for generated midi file
    "-t", "--train": boolean flag if you want to train
    "-g", "--generate": boolean flag if you want to generate a piece
