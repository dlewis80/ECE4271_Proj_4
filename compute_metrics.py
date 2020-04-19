import os
import sys
import argparse
from argparse import RawTextHelpFormatter

description = r'''
This program computes the accuracy of a sound classifier.
The inputs are the directories containing the text files with
the labels for each audio file.
Both the human-labeled directory and the machine-labeled directory
must have the same number of files and with the sames names. An audio file
called example.wav should have an example.txt file in the human-labeled 
directory and an example.txt file in the machine-labeled directory.
The format of each file must be one line per labeled fragment with 
the starting time and the ending time of the fragment in milliseconds
and the label separated by \t or the white space characters. For example:
    1290    8999    car
    12335   22098   talking
    45566   50003   noise
    67999   90010   car
A fragment is considered to be well classified (true positive) if
the detected fragment interval overlaps with the real fragment.
*** This program does not check the input files. Giving two directories with 
files with different names or bad formatted will give unexpected results*** 
'''
parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
parser.add_argument('human_directory', metavar='human_labels_directory', \
                    type=str, help="Directory with the real labels")
parser.add_argument('machine_directory', metavar='machine_labels_directory', \
                    type=str, help="Directory with the classifier labels")

args = parser.parse_args()
human_directory = args.human_directory
machine_directory = args.machine_directory
#human_directory = "label"
#machine_directory = "prediction"
human_files = []
machine_files = []

for file in os.listdir(human_directory):
    if file.endswith(".txt"):
        human_files.append(file)

for file in os.listdir(machine_directory):
    if file.endswith(".txt"):
        machine_files.append(file)

human_files.sort()
machine_files.sort()

if (human_files != machine_files):
    print("Incorrect input directories. Files must have the same name")
    sys.exit()

for i in range(len(human_files)):
    human_files[i] = os.path.join(human_directory, human_files[i])
    machine_files[i] = os.path.join(machine_directory, machine_files[i])

classes_stats = {}
human_times = []
machine_times = []

bird_tp = 0
no_bird_tp = 0
bird_fp = 0
no_bird_fp = 0
bird_cnt = 0
no_bird_cnt = 0

for i in range(len(human_files)):
    human_times.clear()
    machine_times.clear()
    with open(human_files[i]) as fh, open(machine_files[i]) as fm:
        for line in fh:
            tokens = line.split()
            human_times.append([float(tokens[0]), float(tokens[1]), tokens[2].rstrip(), False,False])
        for line in fm:
            tokens = line.split()
            machine_times.append([float(tokens[0]), float(tokens[1]), tokens[2].rstrip(), False,False])

    for k in human_times:
        if (k[2] not in classes_stats.keys()):
            classes_stats[k[2]] = {'false_positives': 0, 'true_positives': 0, "count": 0}

    for l in machine_times:
        if (l[2] not in classes_stats.keys()):
            classes_stats[l[2]] = {'false_positives': 0, 'true_positives': 0, "count": 0}

    for k in human_times:
        for l in machine_times:
            # Checking if two fragments with the same label overlap
            if (k[0] <= l[1] and k[1] >= l[0]):
                if k[2] == l[2]:
                        k[3] = True
                        l[3] = True

                if k[2][0:4] == 'Bird' and l[2][0:4] == 'Bird':
                    k[4] = True
                    l[4] = True
                if k[2][0:4] != 'Bird' and l[2][0:4] != 'Bird':
                    k[4] = True
                    l[4] = True
            

    # Counting the number of overlapping files and stats
    for k in human_times:
        classes_stats[k[2]]["count"] = classes_stats[k[2]]["count"] + 1
        if (k[2][0:4] == 'Bird'):
            bird_cnt = bird_cnt +1
        else:
            no_bird_cnt = no_bird_cnt +1

        if (k[3] == True):
            classes_stats[k[2]]["true_positives"] = classes_stats[k[2]]["true_positives"] + 1
        if (k[4] == True):
            if (k[2][0:4] == 'Bird'):
                bird_tp = bird_tp +1
            else:
                no_bird_tp = no_bird_tp +1

    for l in machine_times:
        if (l[3] == False):
            classes_stats[l[2]]["false_positives"] = classes_stats[l[2]]["false_positives"] + 1
        if (l[4] == False):
            if(l[2][0:4] == 'Bird'):
                bird_fp = bird_fp +1
            else:
                no_bird_fp = no_bird_fp +1

total_n = 0
total_tp = 0
total_fp = 0

for k in classes_stats:
    tp = classes_stats[k]["true_positives"]
    fp = classes_stats[k]["false_positives"]
    cnt = classes_stats[k]["count"]
    total_n = total_n + cnt
    total_tp = total_tp + tp
    total_fp = total_fp + fp
    print("Class: " + k)
    print("    Number of cases: " + str(cnt))
    print("    True positives: " + str(tp))
    print("    False positives: " + str(fp))
    if (cnt != 0):
        print("    TPR: " + str(float(tp / cnt)))
    else:
        print("    TPR: does not exist. There is no fragment with this label in the file. Only false positives")
    print("")

print("Accuracy: " + str(float(total_tp / total_n)))
print("Total number of false positives(in absolute terms): " + str(total_fp))

print("\n")
print("Bird or no bird stats(dividing all classes into two classes: \"Bird\" or \"No Bird\", \"Bird\" is considered to be the positive case\n,  and \"No Bird\" class the negative one):")
print ("")
print("Class: Bird")
print("    Number of cases: "+str(bird_cnt))
print("    True positives: "+str(bird_tp))
print("    False positives: "+str(bird_fp))
print("    TPR: "+str(bird_tp/bird_cnt))
print("Class: No bird")
print("    Number of cases: "+str(no_bird_cnt))
print("    True positives: "+str(no_bird_tp))
print("    False negatives: "+str(no_bird_fp))
print("    TNR: "+str(no_bird_tp/no_bird_cnt))
print("\n")
print('Number of cases: '+str(bird_cnt+no_bird_cnt))
print('Correctly detected fragments: '+str(bird_tp+no_bird_tp))
print('False positives + False Negatives: '+str(bird_fp+no_bird_fp))
print('Accuracy: '+str(float((bird_tp+no_bird_tp)/(bird_cnt+no_bird_cnt))))
print("")
caut_message = '''
************************************************************************
Take into account that the Total number of False Positives number is in absolute value,
which means that if the classifier labels a few seconds incorrectly,
if the predicted labels refer to small window sizes, the False positive number 
will be big (even if the classifier works good). If the windows overlap, then the False Positives number
will be even bigger. Because we are approximating the detection problem with overlapping time intervals, there
could be a True positive and a False positive at the same time for one fragment because one 
human-labeled fragment could have several machine labeled fragments overlapping with it.
A better analysis of the false positives will be added to this script if time allows.
************************************************************************ 
'''
print(caut_message)
print("")
