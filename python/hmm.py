import numpy as np

class HMM:
    """
    HMM class used to compose songs.
    """

    def __init__(self, numObsVars=0, numHiddenStates=0, numPossibleObs=0, filePath=None, obsDictFilePath=None):
        self.a = None               # transition prob matrix, M x M
        self.b = None               # emission prob matrix, M x K x R
        self.pi = None              # initial state probs, 1 x M x R
        self.numPossibleObs = numPossibleObs    # K
        self.numHiddenStates = numHiddenStates  # M
        self.numObservationVars = numObsVars    # R = numObsVar + numTrack
        self.alphas = None          # M x R x T
        self.betas = None           # M x R x T
        self.seqLength = 0

        if filePath and obsDictFilePath is not None:
            self.loadModel(filePath, obsDictFilePath)

    def initialize(self, transitionProbs=None, emissionProbs=None, initialState=None, obsDict=None):
        if transitionProbs is None:
            a = np.random.random((self.numHiddenStates, self.numHiddenStates))
            self.a = a / np.sum(a, axis=0)
        else:
            self.a = transitionProbs
            self.numHiddenStates = transitionProbs.shape[0]

        if emissionProbs is None:
            b = np.random.random((self.numHiddenStates, self.numPossibleObs, self.numObservationVars))
            print(self.numObservationVars)
            self.b = b / np.sum(b, axis=0)
        else:
            self.b = emissionProbs
            self.numPossibleObs = emissionProbs.shape[1]
            self.numObservationVars = emissionProbs.shape[2]

        if initialState is None:
            pi = np.random.random((self.numHiddenStates))
            self.pi = pi / np.sum(pi)
        else:
            self.pi = initialState

        if obsDict is None:
            self.obsDict = None
        else:
            self.obsDict = obsDict


    def saveModel(self, filePath, obsDictFilePath):
        np.savez(filePath, initState=self.pi, emitMat=self.b, transMat=self.a)
        print("model saved to " + filePath)
        newObsDict = {}
        for key in self.obsDict.keys():
            newObsDict[str(self.obsDict[key])] = key
        np.savez(obsDictFilePath, **newObsDict)
        print("obsDict saved to " + filePath)


    def loadModel(self, filePath, obsDictFilePath):
        obsDict = {}
        with np.load(obsDictFilePath, allow_pickle=True) as obsFile:
            for f in obsFile.files:
                obsDict[int(f)] = tuple(obsFile[f])
        with np.load(filePath, allow_pickle=True) as modelFile:
            self.initialize(modelFile['transMat'], modelFile['emitMat'], modelFile['initState'], obsDict)


    def train(self, songData, threshold=1e-6, maxEpochs=50, filePath=None, obsDictFilePath=None):
        # T x R x vars
        obsSequence = songData.getShortenedTracks(90)
        self.obsDict = songData.getAllPossibleObservations()
        self.seqLength = len(obsSequence)
        epoch = 0
        prevScore = 0
        scoreDelta = 1

        while epoch < maxEpochs and abs(scoreDelta) > threshold:
            eta, xi, pO_model = self.eStep(obsSequence)
            score = self.mStep(eta, xi, pO_model, obsSequence)
            scoreDelta = score - prevScore
            prevScore = score
            epoch += 1
            print("Training HMM. Epoch " + str(epoch) + ": " + str(score))

        if filePath is not None and obsDictFilePath is not None:
            self.saveModel(filePath, obsDictFilePath)


    def generateSequence(self, length):
        """
        Generate a sequence of the given length
        """
        # sample initial state
        curState = np.random.choice(range(self.numHiddenStates), p=self.pi)
        tracks = np.empty((length, self.numObservationVars, 5), dtype=int)

        # generate first observation
        o0 = []
        for x in range(self.numObservationVars):
            bProbs = self.b[curState, :, x]
            output = np.random.choice(range(self.numPossibleObs), p=bProbs/np.sum(bProbs))
            o0.append(output)
            tracks[0, x] = self.obsDict[output]
        prevState = curState

        # sample from transition/emission matrix for each successive state/emission
        for t in range(1, length):
            # sample from transition for new state
            curState = np.random.choice(range(self.numHiddenStates), p=self.a[prevState]/np.sum(self.a[prevState]))

            for x in range(self.numObservationVars):
                bProbs = self.b[curState, :, x]
                output = np.random.choice(range(self.numPossibleObs), p=bProbs / np.sum(bProbs))
                o0.append(output)
                tracks[t, x] = self.obsDict[output]
            prevState = curState

        return tracks.reshape((self.numObservationVars, length,  5))

    def calcAlphas(self, obSequence):
        """
        calculates alphas given the observation sequence.
        """
        alphas = np.zeros((self.numHiddenStates, self.numObservationVars, self.seqLength))

        # initialize first
        for x in range(self.numObservationVars):
            bVal = self.obsDict[obSequence[0][x]]
            alphas[:, x, 0] = np.multiply(self.pi, self.b[:, bVal, x])

        for t in range(1, self.seqLength):
            for s in range(self.numHiddenStates):

                # 1 x R x 1             # M x R x 1        # M x 1       # M x R
                for x in range(self.numObservationVars):
                    bVal = self.obsDict[obSequence[t][x]]
                    alphas[:, x, t] = np.sum(alphas[:, x, t - 1] * self.a[:, s] * self.b[:, bVal, x], axis=0)

        return alphas


    def calcBetas(self, obSequence):
        """
        calculates betas given the observation sequence.
        """
        # beta_t(i) = P[O_t+1:T | Z_t, Theta] = sum_Z_t+1 ( beta_t+1(j) * P(Z_t+1 = j | Z_t) * P(O_t+1 | Z_t+1) )
        # beta_t(i) = P[O_t+1:T | Z_t, Theta] = sum_Z_t+1 ( beta_t+1(j) * a[i,j] * b[j](O_t+1) * P(O_t+1 | Z_t+1) )
        betas = np.zeros((self.numHiddenStates, self.numObservationVars, self.seqLength))

        # initialize end
        betas[:, :, self.seqLength-1] = np.ones((self.numHiddenStates, self.numObservationVars))

        for t in reversed(range(self.seqLength-1)):
            for s in range(self.numHiddenStates):
                for x in range(self.numObservationVars):
                    bVal = self.obsDict[obSequence[t + 1][x]]
                    betas[:, x, t] = np.sum(betas[:, x, t + 1] * self.a[:, s] * self.b[:, bVal, x], axis=0)

        return betas

    def eStep(self, obSequence):
        """
        Performs the first step of baum welch
        :return:
        """

        self.alphas = self.calcAlphas(obSequence)  # M x R x T
        self.betas = self.calcBetas(obSequence)  # M x R x T

        # normalize alphas and betas
        normFactor = np.sum(self.alphas, axis=0)
        if np.all(normFactor > 0):
            self.alphas = (self.alphas) / normFactor
            self.betas = (self.betas) / normFactor

        # probability of state i at time t
        # eta.shape = M x R x T
        eta = self.alphas * self.betas

        # 1 x R x T
        # pO_model = np.sum(np.sum(eta, axis=0), axis=0)
        pO_model = np.sum(eta, axis=0)

        eta = eta/pO_model

        # probability of transition from state i to j at time t
        # xi.shape = M x M x R x T
        xi = np.zeros((self.numHiddenStates, self.numHiddenStates, self.numObservationVars, self.seqLength))
        for t in range(self.seqLength - 1):
            for s in range(self.numHiddenStates):
                # M x R x 1     # M x R x 1         # M x R x 1     # M x M     # M x 1
                for x in range(self.numObservationVars):
                    bVal = self.obsDict[obSequence[t][x]]
                    xi[:, s, x, t] = self.alphas[:, x, t] * self.betas[:, x, t] * self.a[:, s] * \
                                     self.b[:, bVal, x]

        xi = xi/np.sum(np.sum(xi, axis=1),axis=0)

        return eta, xi, normFactor



    def mStep(self, eta, xi, pO_model, obSequence):
        """
        performs the maximization/update step of baum welch
        :return:
        """

        # sum over each observation var to get new probabilities for starting states
        newPi = np.sum(eta[:,:,0], axis=1)/self.numObservationVars
        self.pi = newPi

        newDenom = np.sum(eta, axis=2)

        # Expected # of transitions from i = sum_t ( eta_t(i) )
        # newA[i,j] = ( sum_t:t-1 ( xi_t(i,j) ) ) / ( sum_t ( T - eta_t(j) ) )
        denominator = np.sum(eta[:, :, :self.seqLength-1], axis=2)
        self.a = np.sum(np.sum(xi[:, :, :, :self.seqLength-1], axis=3)/denominator, axis=2)


        # emission matrix updates
        # expected # transitions from i to j = sum_t ( xi_t(i,j) )
        bIndicator = np.zeros((self.numHiddenStates, self.numPossibleObs, self.numObservationVars, self.seqLength))
        for t in range(self.seqLength):
            for x in range(self.numObservationVars):
                bVal = self.obsDict[obSequence[t][x]]
                bIndicator[:, bVal, :, t] = eta[:, :, t]

        self.b = np.sum(bIndicator, axis=3)/np.sum(newDenom)

        return np.sum(pO_model)
