from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import MidTermFeatures
import numpy as np
import pickle
from sklearn.svm import SVC

#dictionary to store labels
labels = {}
features = {}

#load features
# pkl_file = open('learned_features.pkl', 'rb')
# features = pickle.load(pkl_file)
# pkl_file.close()

#Process Audacity generated labelled file assumes files are names in format 5E6BA3C8_labelled.txt
#Inputs : filePath, the path name to the audacity label file
#Outputs: None
def process_label(filePath):
    audioName = filePath[-21:]
    audioName = audioName.split("_")[0]
    labels.update({audioName:{}})
    with open(filePath, "r") as labelFile:
        label = labelFile.readline()
        while label:
            #split line into three values
            labelData = label.split()
            startTime = float(labelData[0])
            endTime = float(labelData[1])
            soundType = labelData[2]
            #access dictionary corresponding to this audio file
            thisAudio = labels[audioName]
            #check if soundType is present and modify dictionary
            if soundType in thisAudio:
                thisAudio[soundType].append([startTime,endTime])
            else:
                thisAudio.update({soundType:[[startTime,endTime]]})
            #add sound type to feature dictionary
            if soundType not in features:
                features.update({soundType:[]})
            #read next line for processing
            label = labelFile.readline()

#Cross-references a specific time window with the audioLabels to find intersection, returns which soundTypes intersect
#with this window
#Inputs: window, window[0] is the start time of the window and window[1] is the end time
#Outputs intersections, array of strings representing sound types that correspond to audio window
def sound_type_lookup(window, audioLabels):
    soundTypes = audioLabels.keys()
    intersections = []
    for soundType in soundTypes:
        times = audioLabels[soundType]
        for time in times:
            startTime = time[0]
            endTime = time[1]
            #check for intersection
            if window[0] < endTime and window[1] > startTime:
                intersections.append(soundType)
    return intersections

#Computes feature sets for audio file. 34 features are computed they include: Zero Crossing Rate, Energy, Entropy of
#Energy, Spectral Centroid, Spectral Spread, Spectral Entropy, Spectral Flux, Spectral Rolloff, MFCCs (9-21), Chroma
#vector (22-33), and Chroma Deviation. In addition the deltas (difference between current and last feature vector) are
#computed. These are the short term features (64 in total), the mid term features are the mean and variance statistics
#of the short term features. Together they form 204 total features (64 short term, 64 mean, 64 variance)
#These features are compiled into matrix of dimension 204xN where N is the number windows that fit in audio file.
#Then each column of features is assigned to a specific sound type and placed in global feature dictionary.
#Inputs: labelsDict: dictionary of labels
#        audioFilePath: path to audio file
#        windowSize: specifies size of window in seconds
#        windowStep: specifies size of step in seconds
#Outputs: None
def process_learning_features(labelsDict, audioFilePath, windowSize, windowStep):
    audioName = audioFilePath[-12:]
    audioName = audioName.split(".")[0]
    [Fs, signal] = audioBasicIO.read_audio_file(audioFilePath)
    mF, sF, f_names = MidTermFeatures.mid_feature_extraction(signal, Fs, windowSize * Fs, windowStep * Fs, \
                                                             windowSize * Fs, windowStep * Fs)
    #Window that corresponds to a single column in feature array F
    timeWindow = [0.0,windowSize]
    #look at labels for this audio file
    thisAudio = labelsDict[audioName]
    count = 0
    while count < np.size(mF,1):
        intersections = sound_type_lookup(timeWindow, thisAudio)
        for inter in intersections:
            features[inter].append(np.concatenate((sF[:,count], mF[:,count])))
        count += 1
        timeWindow[0] += windowStep
        timeWindow[1] += windowStep

#This function converts a dictionary of features into a matrix to be used for training the SVM machine
#Inputs: featuresDict: Dictionary of features where the keys are the name of the audio file processed and the values are
def dict_to_training_matrix(featuresDict):
    classes = features.keys()
    count = 0
    matrix = np.ndarray((0,204))
    classLabels = []
    for cl in classes:
        thisClass = features[cl]
        numRows = np.size(thisClass,0)
        numCols = np.size(thisClass,1)
        ftMatrix = np.zeros((numRows,numCols))
        for i in range(0,numRows):
            ftMatrix[i,0:numCols] = thisClass[i]
            classLabels.append(count)
        matrix = np.concatenate((matrix,ftMatrix))
        count += 1
    return matrix, np.transpose(np.asarray(classLabels))

#Computes feature set for audio file to be classified
#Inputs:  audioPath: Path to audio file to be classified
#         windowSize: Size of window in seconds to compute features for
#         windowStep: Step size of window, overlaps if less than windowSize
#Outputs: F: Matrix of features for audio file
def test_features(audioPath, windowSize, windowStep):
    [Fs, signal] = audioBasicIO.read_audio_file(audioPath)
    mF, sF, f_names = MidTermFeatures.mid_feature_extraction(signal, Fs, windowSize * Fs, windowStep * Fs, \
                                                             windowSize * Fs, windowStep * Fs)
    F = np.vstack((sF, mF))
    F = np.transpose(F)
    return F

#Takes output of SVM classifier and converts it to equivalent format of audacity generated label files
#Inputs:  y_pred: (vector) The output of the SVM classifier that assigns a class to each time window in audio file
#         fileName: (string) Name of the file to write text to
#         windowSize: Size of window in seconds to compute features for
#         windowStep: Step size of window, overlaps if less than windowSize
#Outputs: None
def interpret_prediction(y_pred, fileName, windowSize, windowStep):
    file = open(fileName, 'w+')
    timeWindow = [0.0, windowSize]
    classes = list(features.keys())
    for i in range(0,len(y_pred)):
        file.write("%9.6f	%9.6f	%s\n" % (timeWindow[0], timeWindow[1], classes[y_pred[i]]))
        timeWindow[0]+=windowStep
        timeWindow[1]+=windowStep

#Takes output text file from interpret_prediction and condenses consecutive labels of the same class to a single time
#stamp
#Inputs:  filePath: path to file of un-condensed labels
#         newFilePath: path for new file with condensed labels
#Outputs: None
def condense_labels(filePath, newFilePath):
    file = open(newFilePath, 'w+')
    with open(filePath, "r") as labelFile:
        label = labelFile.readline()
        labelData = label.split()
        start = float(labelData[0])
        end = float(labelData[1])
        currentLabel = []
        previousLabel = labelData[2]
        label = labelFile.readline()
        if not label:
            file.write("%9.6f	%9.6f	%s\n" % (start, end, previousLabel))
        while label:
            labelData = label.split()
            currentLabel = labelData[2]
            if currentLabel == previousLabel:
                end = float(labelData[1])
            elif currentLabel != previousLabel:
                file.write("%9.6f	%9.6f	%s\n" % (start, end, previousLabel))
                start = float(labelData[0])
                end = float(labelData[1])
            previousLabel = currentLabel
            label = labelFile.readline()

process_label("/Users/omarwali/ece4271/project4/label_files/anderson/5E6BA3C8_labelled.txt")
process_label("/Users/omarwali/ece4271/project4/label_files/anderson/5E6BA406_labelled.txt")
process_label("/Users/omarwali/ece4271/project4/label_files/anderson/5E6BA444_labelled.txt")
process_label("/Users/omarwali/ece4271/project4/label_files/anderson/5E6C7D44_labelled.txt")
process_label("/Users/omarwali/ece4271/project4/label_files/anderson/5E6CB992_labelled.txt")
process_label("/Users/omarwali/ece4271/project4/label_files/anderson/5E6BD3FA_labelled.txt")
process_label("/Users/omarwali/ece4271/project4/label_files/anderson/5E6BD2C4_labelled.txt")
process_learning_features(labels,"/Users/omarwali/ece4271/project4/Anderson-serviceberry-selected/5E6BA3C8.WAV", 0.25, 0.025)
process_learning_features(labels,"/Users/omarwali/ece4271/project4/Anderson-serviceberry-selected/5E6BA406.WAV", 0.25, 0.025)
process_learning_features(labels,"/Users/omarwali/ece4271/project4/Anderson-serviceberry-selected/5E6BA444.WAV", 0.25, 0.025)
process_learning_features(labels,"/Users/omarwali/ece4271/project4/Anderson-serviceberry-selected/5E6C7D44.WAV", 0.25, 0.025)
process_learning_features(labels,"/Users/omarwali/ece4271/project4/Anderson-serviceberry-selected/5E6CB992.WAV", 0.25, 0.025)
process_learning_features(labels,"/Users/omarwali/ece4271/project4/Anderson-serviceberry-selected/5E6BD3FA.WAV", 0.25, 0.025)
process_learning_features(labels,"/Users/omarwali/ece4271/project4/Anderson-serviceberry-selected/5E6BD2C4.WAV", 0.25, 0.025)

output = open('learned_features.pkl', 'wb')
pickle.dump(features, output)
output.close()

matrix, classLabels = dict_to_training_matrix(features)
svclassifier = SVC(kernel='linear')
svclassifier.fit(matrix, classLabels)

x_test = test_features("/Users/omarwali/ece4271/project4/Anderson-serviceberry-selected/5E6BB8DA.WAV", 0.25, 0.025)
y_pred = svclassifier.predict(x_test)
interpret_prediction(y_pred,"5E6BB8DA_labelled_UC.txt",0.25,0.025)
condense_labels("5E6BB8DA_labelled_UC.txt", "5E6BB8DA_labelled.txt")
