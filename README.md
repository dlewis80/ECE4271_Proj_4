# ECE4271_Proj_4
Audio classification project for ECE4271 at GT

To run classifier on audio file copy the path to the file and run:

python3 proj4.py /audioFilePath/

Output will be in format AudioName_labelled.txt and will be in the machine_labels directory.

To compute metrics place human labelled files in a folder and machine labelled files in another and run:

python3 compute_metrics.py human_labels/ machine_labels/
