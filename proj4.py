from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import MidTermFeatures
import numpy as np
from sklearn.svm import SVC

#dictionary to store labels
labels = {}
features = {}

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

#Computes feature sets for audio file. 34 features are computed and compiled into matrix of dimension 34xN where N is
#the number windows that fit in audio file. Then each column of features is assigned to a specific sound type and
#placed in global feature dictionary
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

def test_features(audioPath, windowSize, windowStep):
    [Fs, signal] = audioBasicIO.read_audio_file(audioPath)
    mF, sF, f_names = MidTermFeatures.mid_feature_extraction(signal, Fs, windowSize * Fs, windowStep * Fs, \
                                                             windowSize * Fs, windowStep * Fs)
    F = np.vstack((sF, mF))
    F = np.transpose(F)
    return F

def interpret_prediction(y_pred, fileName, windowSize, windowStep):
    file = open(fileName, 'w+')
    timeWindow = [0.0, windowSize]
    classes = list(features.keys())
    for i in range(0,len(y_pred)):
        file.write("%9.6f	%9.6f	%s\n" % (timeWindow[0], timeWindow[1], classes[y_pred[i]]))
        timeWindow[0]+=windowStep
        timeWindow[1]+=windowStep


process_label("/Users/omarwali/ece4271/project4/label_files/anderson/5E6BA3C8_labelled.txt")
process_label("/Users/omarwali/ece4271/project4/label_files/anderson/5E6BA406_labelled.txt")
process_label("/Users/omarwali/ece4271/project4/label_files/anderson/5E6BA444_labelled.txt")
process_label("/Users/omarwali/ece4271/project4/label_files/anderson/5E6C7D44_labelled.txt")
process_label("/Users/omarwali/ece4271/project4/label_files/anderson/5E6CB992_labelled.txt")
process_learning_features(labels,"/Users/omarwali/ece4271/project4/Anderson-serviceberry-selected/5E6BA3C8.WAV", 0.25, 0.025)
process_learning_features(labels,"/Users/omarwali/ece4271/project4/Anderson-serviceberry-selected/5E6BA406.WAV", 0.25, 0.025)
process_learning_features(labels,"/Users/omarwali/ece4271/project4/Anderson-serviceberry-selected/5E6BA444.WAV", 0.25, 0.025)
process_learning_features(labels,"/Users/omarwali/ece4271/project4/Anderson-serviceberry-selected/5E6C7D44.WAV", 0.25, 0.025)
process_learning_features(labels,"/Users/omarwali/ece4271/project4/Anderson-serviceberry-selected/5E6CB992.WAV", 0.25, 0.025)

matrix, classLabels = dict_to_training_matrix(features)
svclassifier = SVC(kernel='linear')
svclassifier.fit(matrix, classLabels)

x_test = test_features("/Users/omarwali/ece4271/project4/Anderson-serviceberry-selected/5E6BA482.WAV", 0.25, 0.025)
y_pred = svclassifier.predict(x_test)
interpret_prediction(y_pred,"5E6BA42_prediction.txt",0.25,0.025)
print(y_pred)
