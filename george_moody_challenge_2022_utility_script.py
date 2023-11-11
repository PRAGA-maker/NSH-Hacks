#!/usr/bin/env python

# Do *not* edit this script.
# These are helper functions that you can use with your code.

import os, numpy as np, scipy as sp, scipy.io, scipy.io.wavfile
import os.path, sys


# Check if a variable is a number or represents a number.
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

# Check if a variable is an integer or represents an integer.
def is_integer(x):
    if is_number(x):
        return float(x).is_integer()
    else:
        return False

# Check if a variable is a a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False

# Compare normalized strings.
def compare_strings(x, y):
    try:
        return str(x).strip().casefold()==str(y).strip().casefold()
    except AttributeError: # For Python 2.x compatibility
        return str(x).strip().lower()==str(y).strip().lower()

# OG
def find_patient_files(data_folder):
    # Find patient files.
    filenames = list()
    for f in sorted(os.listdir(data_folder)):
        root, extension = os.path.splitext(f)
        if not root.startswith('.') and extension=='.txt':
            filename = os.path.join(data_folder, f)
            filenames.append(filename)

    # To help with debugging, sort numerically if the filenames are integers.
    roots = [os.path.split(filename)[1][:-4] for filename in filenames]
    if all(is_integer(root) for root in roots):
        filenames = sorted(filenames, key=lambda filename: int(os.path.split(filename)[1][:-4]))

    return filenames


# Edited -- Find patient data .wav files.
def find_patient_wav_files(data_folder):
    # Find patient files.
    filenames = list()
    for f in sorted(os.listdir(data_folder)):
        root, extension = os.path.splitext(f)
        if not root.startswith('.') and extension=='.wav': #normally .txt
            filename = os.path.join(data_folder, f)
            filenames.append(filename)

    # To help with debugging, sort numerically if the filenames are integers.
    roots = [os.path.split(filename)[1][:-4] for filename in filenames]
    if all(is_integer(root) for root in roots):
        filenames = sorted(filenames, key=lambda filename: int(os.path.split(filename)[1][:-4]))

    return filenames

#Edited -- Find all patient data .txt files
def find_patient_txt_files(data_folder):
    # Find patient files.
    filenames = list()
    for f in sorted(os.listdir(data_folder)):
        root, extension = os.path.splitext(f)
        if not root.startswith('.') and extension=='.txt': 
            filename = os.path.join(data_folder, f)
            filenames.append(filename)

    # To help with debugging, sort numerically if the filenames are integers.
    roots = [os.path.split(filename)[1][:-4] for filename in filenames]
    if all(is_integer(root) for root in roots):
        filenames = sorted(filenames, key=lambda filename: int(os.path.split(filename)[1][:-4]))
    return filenames

# Load patient data as a string.
def load_patient_data(filename):
    with open(filename, 'r') as f:
        data = f.read()
    return data

# Load a WAV file.
def load_wav_file(filename):
    frequency, recording = sp.io.wavfile.read(filename)
    return recording, frequency

# Load recordings.
def load_recordings(data_folder, data, get_frequencies=False):
    num_locations = get_num_locations(data)
    recording_information = data.split('\n')[1:num_locations+1]

    recordings = list()
    frequencies = list()
    for i in range(num_locations):
        entries = recording_information[i].split(' ')
        recording_file = entries[2]
        filename = os.path.join(data_folder, recording_file)
        recording, frequency = load_wav_file(filename)
        recordings.append(recording)
        frequencies.append(frequency)

    if get_frequencies:
        return recordings, frequencies
    else:
        return recordings

# Get patient ID from patient data.
def get_patient_id(data):
    patient_id = None
    for i, l in enumerate(data.split('\n')):
        if i==0:
            try:
                patient_id = l.split(' ')[0]
            except:
                pass
        else:
            break
    return patient_id

# Get number of recording locations from patient data.
def get_num_locations(data):
    num_locations = None
    for i, l in enumerate(data.split('\n')):
        if i==0:
            try:
                num_locations = int(l.split(' ')[1])
            except:
                pass
        else:
            break
    return num_locations

# Get frequency from patient data.
def get_frequency(data):
    frequency = None
    for i, l in enumerate(data.split('\n')):
        if i==0:
            try:
                frequency = float(l.split(' ')[2])
            except:
                pass
        else:
            break
    return frequency

# Get recording locations from patient data.
def get_locations(data):
    num_locations = get_num_locations(data)
    locations = list()
    for i, l in enumerate(data.split('\n')):
        entries = l.split(' ')
        if i==0:
            pass
        elif 1<=i<=num_locations:
            locations.append(entries[0])
        else:
            break
    return locations

# Get age from patient data.
def get_age(data):
    age = None
    for l in data.split('\n'):
        if l.startswith('#Age:'):
            try:
                age = l.split(': ')[1].strip()
            except:
                pass
    return age

# Get sex from patient data.
def get_sex(data):
    sex = None
    for l in data.split('\n'):
        if l.startswith('#Sex:'):
            try:
                sex = l.split(': ')[1].strip()
            except:
                pass
    return sex

# Get height from patient data.
def get_height(data):
    height = None
    for l in data.split('\n'):
        if l.startswith('#Height:'):
            try:
                height = float(l.split(': ')[1].strip())
            except:
                pass
    return height

# Get weight from patient data.
def get_weight(data):
    weight = None
    for l in data.split('\n'):
        if l.startswith('#Weight:'):
            try:
                weight = float(l.split(': ')[1].strip())
            except:
                pass
    return weight

# Get pregnancy status from patient data.
def get_pregnancy_status(data):
    is_pregnant = None
    for l in data.split('\n'):
        if l.startswith('#Pregnancy status:'):
            try:
                is_pregnant = bool(sanitize_binary_value(l.split(': ')[1].strip()))
            except:
                pass
    return is_pregnant

# Get labels from patient data.
def get_label(data):
    label = None
    for l in data.split('\n'):
        if l.startswith('#Murmur:'):
            try:
                label = l.split(': ')[1]
            except:
                pass
    if label is None:
        raise ValueError('No label available. Is your code trying to load labels from the hidden data?')
    return label

#Systolic murmur pitch: nan
#Systolic murmur quality: nan
#Diastolic murmur timing: nan
#Diastolic murmur shape: nan
#Diastolic murmur grading: nan
#Diastolic murmur pitch: nan

def get_pitch_label(data):
    label = None
    for l in data.split('\n'):
        if l.startswith('#Systolic murmur pitch:') or l.startswith('#Diastolic murmur pitch:'):
            try:
                label = l.split(': ')[1]
            except:
                pass
    if label is None:
        raise ValueError('No label available. Is your code trying to load labels from the hidden data?')
    return label

# Sanitize binary values from Challenge outputs.
def sanitize_binary_value(x):
    x = str(x).replace('"', '').replace("'", "").strip() # Remove any quotes or invisible characters.
    if (is_finite_number(x) and float(x)==1) or (x in ('True', 'true', 'T', 't')):
        return 1
    else:
        return 0

# Santize scalar values from Challenge outputs.
def sanitize_scalar_value(x):
    x = str(x).replace('"', '').replace("'", "").strip() # Remove any quotes or invisible characters.
    if is_finite_number(x) or (is_number(x) and np.isinf(float(x))):
        return float(x)
    else:
        return 0.0

# Save Challenge outputs.
def save_challenge_outputs(filename, patient_id, classes, labels, probabilities):
    # Format Challenge outputs.
    patient_string = '#{}'.format(patient_id)
    class_string = ','.join(str(c) for c in classes)
    label_string = ','.join(str(l) for l in labels)
    probabilities_string = ','.join(str(p) for p in probabilities)
    output_string = patient_string + '\n' + class_string + '\n' + label_string + '\n' + probabilities_string + '\n'

    # Write the Challenge outputs.
    with open(filename, 'w') as f:
        f.write(output_string)

# Load Challenge outputs.
def load_challenge_outputs(filename):
    with open(filename, 'r') as f:
        for i, l in enumerate(f):
            if i==0:
                patient_id = l.replace('#', '').strip()
            elif i==1:
                classes = tuple(entry.strip() for entry in l.split(','))
            elif i==2:
                labels = tuple(sanitize_binary_value(entry) for entry in l.split(','))
            elif i==3:
                probabilities = tuple(sanitize_scalar_value(entry) for entry in l.split(','))
            else:
                break
    return patient_id, classes, labels, probabilities











# Check if a variable is a number or represents a number.
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

# Check if a variable is an integer or represents an integer.
def is_integer(x):
    if is_number(x):
        return float(x).is_integer()
    else:
        return False

# Check if a variable is a a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False

# Compare normalized strings.
def compare_strings(x, y):
    try:
        return x.strip().casefold()==y.strip().casefold()
    except AttributeError: # For Python 2.x compatibility
        return x.strip().lower()==y.strip().lower()

# Edited -- Find patient data .wav files.
def find_patient_wav_files(data_folder):
    # Find patient files.
    filenames = list()
    for f in sorted(os.listdir(data_folder)):
        root, extension = os.path.splitext(f)
        if not root.startswith('.') and extension=='.wav': #normally .txt
            filename = os.path.join(data_folder, f)
            filenames.append(filename)

    # To help with debugging, sort numerically if the filenames are integers.
    roots = [os.path.split(filename)[1][:-4] for filename in filenames]
    if all(is_integer(root) for root in roots):
        filenames = sorted(filenames, key=lambda filename: int(os.path.split(filename)[1][:-4]))

    return filenames

#Edited -- Find all patient data .txt files
def find_patient_txt_files(data_folder):
    # Find patient files.
    filenames = list()
    for f in sorted(os.listdir(data_folder)):
        root, extension = os.path.splitext(f)
        if not root.startswith('.') and extension=='.txt': 
            filename = os.path.join(data_folder, f)
            filenames.append(filename)

    # To help with debugging, sort numerically if the filenames are integers.
    roots = [os.path.split(filename)[1][:-4] for filename in filenames]
    if all(is_integer(root) for root in roots):
        filenames = sorted(filenames, key=lambda filename: int(os.path.split(filename)[1][:-4]))

    return filenames

# Load patient data as a string.
def load_patient_data(filename):
    with open(filename, 'r') as f:
        data = f.read()
    return data

# Load a WAV file.
def load_wav_file(filename):
    frequency, recording = sp.io.wavfile.read(filename)
    return recording, frequency

# Load recordings.
def load_recordings(data_folder, data, get_frequencies=False):
    num_locations = get_num_locations(data)
    recording_information = data.split('\n')[1:num_locations+1]

    recordings = list()
    frequencies = list()
    for i in range(num_locations):
        entries = recording_information[i].split(' ')
        recording_file = entries[2]
        filename = os.path.join(data_folder, recording_file)
        recording, frequency = load_wav_file(filename)
        recordings.append(recording)
        frequencies.append(frequency)

    if get_frequencies:
        return recordings, frequencies
    else:
        return recordings

# Get patient ID from patient data.
def get_patient_id(data):
    patient_id = None
    for i, l in enumerate(data.split('\n')):
        if i==0:
            try:
                patient_id = l.split(' ')[0]
            except:
                pass
        else:
            break
    return patient_id

# Get number of recording locations from patient data.
def get_num_locations(data):
    num_locations = None
    for i, l in enumerate(data.split('\n')):
        if i==0:
            try:
                num_locations = int(l.split(' ')[1])
            except:
                pass
        else:
            break
    return num_locations

# Get recording locations from patient data.
def get_locations(data):
    num_locations = get_num_locations(data)
    locations = list()
    for i, l in enumerate(data.split('\n')):
        entries = l.split(' ')
        if i==0:
            pass
        elif 1<=i<=num_locations+1:
            locations.append(entries[0])
        else:
            break
    return locations

# Get frequency from patient data.
def get_frequency(data):
    frequency = None
    for i, l in enumerate(data.split('\n')):
        if i==0:
            try:
                frequency = float(l.split(' ')[1])
            except:
                pass
        else:
            break
    return frequency

# Get age from patient data.
def get_age(data):
    age = None
    for l in data.split('\n'):
        if l.startswith('#Age:'):
            try:
                age = l.split(': ')[1].strip()
            except:
                pass
    return age

# Get sex from patient data.
def get_sex(data):
    sex = None
    for l in data.split('\n'):
        if l.startswith('#Sex:'):
            try:
                sex = l.split(': ')[1].strip()
            except:
                pass
    return sex

# Get height from patient data.
def get_height(data):
    height = None
    for l in data.split('\n'):
        if l.startswith('#Height:'):
            try:
                height = float(l.split(': ')[1].strip())
            except:
                pass
    return height

# Get weight from patient data.
def get_weight(data):
    weight = None
    for l in data.split('\n'):
        if l.startswith('#Weight:'):
            try:
                weight = float(l.split(': ')[1].strip())
            except:
                pass
    return weight

# Get pregnancy status from patient data.
def get_pregnancy_status(data):
    is_pregnant = None
    for l in data.split('\n'):
        if l.startswith('#Pregnancy status:'):
            try:
                is_pregnant = bool(l.split(': ')[1].strip())
            except:
                pass
    return is_pregnant

# Get labels from patient data.
def get_label(data):
    label = None
    for l in data.split('\n'):
        if l.startswith('#Murmur:'):
            try:
                label = l.split(': ')[1]
            except:
                pass
    return label

# Sanitize binary values from Challenge outputs.
def sanitize_binary_value(x):
    x = x.replace('"', '').replace("'", "").strip() # Remove any quotes or invisible characters.
    if (is_finite_number(x) and float(x)==1) or (x in ('True', 'true', 'T', 't')):
        return 1
    else:
        return 0

# Santize scalar values from Challenge outputs.
def sanitize_scalar_value(x):
    x = x.replace('"', '').replace("'", "").strip() # Remove any quotes or invisible characters.
    if is_finite_number(x) or (is_number(x) and (float(x)==float('inf') or float(x)==-float('inf'))):
        return float(x)
    else:
        return 0.0

# Save Challenge outputs.
def save_challenge_outputs(filename, patient_id, classes, labels, probabilities):
    # Format Challenge outputs.
    recording_string = '#{}'.format(patient_id)
    class_string = ','.join(str(c) for c in classes)
    label_string = ','.join(str(l) for l in labels)
    probabilities_string = ','.join(str(p) for p in probabilities)
    output_string = recording_string + '\n' + class_string + '\n' + label_string + '\n' + probabilities_string + '\n'

    # Write the Challenge outputs.
    with open(filename, 'w') as f:
        f.write(output_string)

# Load Challenge outputs.
def load_challenge_outputs(filename):
    with open(filename, 'r') as f:
        for i, l in enumerate(f):
            if i==0:
                patient_id = l[1:] if len(l)>1 else None
            elif i==1:
                classes = tuple(entry.strip() for entry in l.split(','))
            elif i==2:
                labels = tuple(sanitize_binary_value(entry) for entry in l.split(','))
            elif i==3:
                probabilities = tuple(sanitize_scalar_value(entry) for entry in l.split(','))
            else:
                break
    return patient_id, classes, labels, probabilities

# %% [code] {"jupyter":{"outputs_hidden":false}}
#!/usr/bin/env python

# Do *not* edit this script. Changes will be discarded so that we can process the models consistently.

# This file contains functions for evaluating models for the 2022 Challenge. You can run it as follows:
#
#   python evaluate_model.py labels outputs scores.csv class_scores.csv
#
# where 'labels' is a folder containing files with the labels, 'outputs' is a folder containing files with the outputs from your
# model, 'scores.csv' (optional) is a collection of scores for the algorithm outputs, # and 'class_scores.csv' (optional) is a
# collection of per-class scores for the algorithm outputs.
#
# Each label or output file must have the format described on the Challenge webpage. The scores for the algorithm outputs include
# the area under the receiver-operating characteristic curve (AUROC), the area under the recall-precision curve (AUPRC), accuracy,
# macro F-measure, and the Challenge score.


# Evaluate the model.
def evaluate_model(label_folder, output_folder):
    # Define classes.
    classes = ['Present', 'Unknown', 'Absent']

    # Load the label and output files and reorder them, if needed, for consistency.
    label_files, output_files = find_challenge_files(label_folder, output_folder)
    labels = load_labels(label_files, classes)
    binary_outputs, scalar_outputs = load_classifier_outputs(output_files, classes)

    # For each patient, set the 'Unknown' class to positive if no class is positive or if multiple classes are positive.
    labels = enforce_positives(labels, classes, 'Unknown')
    binary_outputs = enforce_positives(binary_outputs, classes, 'Unknown')

    # Evaluate the model by comparing the labels and outputs.
    auroc, auprc, auroc_classes, auprc_classes = compute_auc(labels, scalar_outputs)
    accuracy = compute_accuracy(labels, binary_outputs)
    f_measure, f_measure_classes = compute_f_measure(labels, binary_outputs)
    challenge_score = compute_challenge_score(labels, binary_outputs, classes)

    # Return the results.
    return classes, auroc, auprc, auroc_classes, auprc_classes, accuracy, f_measure, f_measure_classes, challenge_score

# Find Challenge files.
def find_challenge_files(label_folder, output_folder):
    label_files = list()
    output_files = list()
    for label_file in sorted(os.listdir(label_folder)):
        label_file_path = os.path.join(label_folder, label_file) # Full path for label file
        if os.path.isfile(label_file_path) and label_file.lower().endswith('.txt') and not label_file.lower().startswith('.'):
            root, ext = os.path.splitext(label_file)
            output_file = root + '.csv'
            output_file_path = os.path.join(output_folder, output_file) # Full path for corresponding output file
            if os.path.isfile(output_file_path):
                label_files.append(label_file_path)
                output_files.append(output_file_path)
            else:
                raise IOError('Output file {} not found for label file {}.'.format(output_file, label_file))

    if label_files and output_files:
        return label_files, output_files
    else:
        raise IOError('No label or output files found.')

# Load labels from header/label files.
def load_labels(label_files, classes):
    num_patients = len(label_files)
    num_classes = len(classes)

    # Use one-hot encoding for the labels.
    labels = np.zeros((num_patients, num_classes), dtype=np.bool_)

    # Iterate over the patients.
    for i in range(num_patients):
        data = load_patient_data(label_files[i])
        label = get_label(data)
        for j, x in enumerate(classes):
            if compare_strings(label, x):
                labels[i, j] = 1

    return labels

# Load outputs from output files.
def load_classifier_outputs(output_files, classes):
    # The outputs should have the following form:
    #
    # #Record ID
    # class_1, class_2, class_3
    #       0,       1,       1
    #    0.12,    0.34,    0.56
    #
    num_patients = len(output_files)
    num_classes = len(classes)

    # Use one-hot encoding for the outputs.
    binary_outputs = np.zeros((num_patients, num_classes), dtype=np.bool_)
    scalar_outputs = np.zeros((num_patients, num_classes), dtype=np.float64)

    # Iterate over the patients.
    for i in range(num_patients):
        patient_id, patient_classes, patient_binary_outputs, patient_scalar_outputs = load_challenge_outputs(output_files[i])

        # Allow for unordered or reordered classes.
        for j, x in enumerate(classes):
            for k, y in enumerate(patient_classes):
                if compare_strings(x, y):
                    binary_outputs[i, j] = patient_binary_outputs[k]
                    scalar_outputs[i, j] = patient_scalar_outputs[k]

    return binary_outputs, scalar_outputs

# For each patient, set a specific class to positive if no class is positive or multiple classes are positive.
def enforce_positives(outputs, classes, positive_class):
    num_patients, num_classes = np.shape(outputs)
    j = classes.index(positive_class)

    for i in range(num_patients):
        if np.sum(outputs[i, :]) != 1:
            outputs[i, :] = 0
            outputs[i, j] = 1
    return outputs

# Compute a binary confusion matrix, where the columns are expert labels and rows are classifier labels.
def compute_confusion_matrix(labels, outputs):
    assert(np.shape(labels)==np.shape(outputs))
    assert(all(value in (0, 1) for value in np.unique(labels)))
    assert(all(value in (0, 1) for value in np.unique(outputs)))

    num_patients, num_classes = np.shape(labels)

    A = np.zeros((num_classes, num_classes))
    for k in range(num_patients):
        i = np.argmax(outputs[k, :])
        j = np.argmax(labels[k, :])
        A[i, j] += 1

    return A

# Compute binary one-vs-rest confusion matrices, where the columns are expert labels and rows are classifier labels.
def compute_one_vs_rest_confusion_matrix(labels, outputs):
    assert(np.shape(labels)==np.shape(outputs))
    assert(all(value in (0, 1) for value in np.unique(labels)))
    assert(all(value in (0, 1) for value in np.unique(outputs)))

    num_patients, num_classes = np.shape(labels)

    A = np.zeros((num_classes, 2, 2))
    for i in range(num_patients):
        for j in range(num_classes):
            if labels[i, j]==1 and outputs[i, j]==1: # TP
                A[j, 0, 0] += 1
            elif labels[i, j]==0 and outputs[i, j]==1: # FP
                A[j, 0, 1] += 1
            elif labels[i, j]==1 and outputs[i, j]==0: # FN
                A[j, 1, 0] += 1
            elif labels[i, j]==0 and outputs[i, j]==0: # TN
                A[j, 1, 1] += 1

    return A

# Compute macro AUROC and macro AUPRC.
def compute_auc(labels, outputs):
    num_patients, num_classes = np.shape(labels)

    # Compute and summarize the confusion matrices for each class across at distinct output values.
    auroc = np.zeros(num_classes)
    auprc = np.zeros(num_classes)

    for k in range(num_classes):
        # We only need to compute TPs, FPs, FNs, and TNs at distinct output values.
        thresholds = np.unique(outputs[:, k])
        thresholds = np.append(thresholds, thresholds[-1]+1)
        thresholds = thresholds[::-1]
        num_thresholds = len(thresholds)

        # Initialize the TPs, FPs, FNs, and TNs.
        tp = np.zeros(num_thresholds)
        fp = np.zeros(num_thresholds)
        fn = np.zeros(num_thresholds)
        tn = np.zeros(num_thresholds)
        fn[0] = np.sum(labels[:, k]==1)
        tn[0] = np.sum(labels[:, k]==0)

        # Find the indices that result in sorted output values.
        idx = np.argsort(outputs[:, k])[::-1]

        # Compute the TPs, FPs, FNs, and TNs for class k across thresholds.
        i = 0
        for j in range(1, num_thresholds):
            # Initialize TPs, FPs, FNs, and TNs using values at previous threshold.
            tp[j] = tp[j-1]
            fp[j] = fp[j-1]
            fn[j] = fn[j-1]
            tn[j] = tn[j-1]

            # Update the TPs, FPs, FNs, and TNs at i-th output value.
            while i < num_patients and outputs[idx[i], k] >= thresholds[j]:
                if labels[idx[i], k]:
                    tp[j] += 1
                    fn[j] -= 1
                else:
                    fp[j] += 1
                    tn[j] -= 1
                i += 1

        # Summarize the TPs, FPs, FNs, and TNs for class k.
        tpr = np.zeros(num_thresholds)
        tnr = np.zeros(num_thresholds)
        ppv = np.zeros(num_thresholds)
        for j in range(num_thresholds):
            if tp[j] + fn[j]:
                tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
            else:
                tpr[j] = float('nan')
            if fp[j] + tn[j]:
                tnr[j] = float(tn[j]) / float(fp[j] + tn[j])
            else:
                tnr[j] = float('nan')
            if tp[j] + fp[j]:
                ppv[j] = float(tp[j]) / float(tp[j] + fp[j])
            else:
                ppv[j] = float('nan')

        # Compute AUROC as the area under a piecewise linear function with TPR/
        # sensitivity (x-axis) and TNR/specificity (y-axis) and AUPRC as the area
        # under a piecewise constant with TPR/recall (x-axis) and PPV/precision
        # (y-axis) for class k.
        for j in range(num_thresholds-1):
            auroc[k] += 0.5 * (tpr[j+1] - tpr[j]) * (tnr[j+1] + tnr[j])
            auprc[k] += (tpr[j+1] - tpr[j]) * ppv[j+1]

    # Compute macro AUROC and macro AUPRC across classes.
    if np.any(np.isfinite(auroc)):
        macro_auroc = np.nanmean(auroc)
    else:
        macro_auroc = float('nan')
    if np.any(np.isfinite(auprc)):
        macro_auprc = np.nanmean(auprc)
    else:
        macro_auprc = float('nan')

    return macro_auroc, macro_auprc, auroc, auprc

# Compute accuracy.
def compute_accuracy(labels, outputs):
    A = compute_confusion_matrix(labels, outputs)

    if np.sum(A) > 0:
        accuracy = np.sum(np.diag(A)) / np.sum(A)
    else:
        accuracy = float('nan')

    return accuracy

# Compute macro F-measure.
def compute_f_measure(labels, outputs):
    num_patients, num_classes = np.shape(labels)

    A = compute_one_vs_rest_confusion_matrix(labels, outputs)

    f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 0, 0], A[k, 0, 1], A[k, 1, 0], A[k, 1, 1]
        if 2 * tp + fp + fn > 0:
            f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            f_measure[k] = float('nan')

    if np.any(np.isfinite(f_measure)):
        macro_f_measure = np.nanmean(f_measure)
    else:
        macro_f_measure = float('nan')

    return macro_f_measure, f_measure

# Compute Challenge score.
def compute_challenge_score(labels, outputs, classes):
    # Define costs. Better to load these costs from an external file instead of defining them here.
    c_algorithm  =     1 # Cost for algorithmic prescreening.
    c_gp         =   250 # Cost for screening from a general practitioner (GP).
    c_specialist =   500 # Cost for screening from a specialist.
    c_treatment  =  1000 # Cost for treatment.
    c_error      = 10000 # Cost for diagnostic error.
    alpha        =   0.5 # Fraction of murmur unknown cases that are positive.

    num_patients, num_classes = np.shape(labels)

    A = compute_confusion_matrix(labels, outputs)

    idx_positive = classes.index('Present')
    idx_unknown  = classes.index('Unknown')
    idx_negative = classes.index('Absent')

    n_pp = A[idx_positive, idx_positive]
    n_pu = A[idx_positive, idx_unknown ]
    n_pn = A[idx_positive, idx_negative]
    n_up = A[idx_unknown , idx_positive]
    n_uu = A[idx_unknown , idx_unknown ]
    n_un = A[idx_unknown , idx_negative]
    n_np = A[idx_negative, idx_positive]
    n_nu = A[idx_negative, idx_unknown ]
    n_nn = A[idx_negative, idx_negative]

    n_total = n_pp + n_pu + n_pn \
        + n_up + n_uu + n_un \
        + n_np + n_nu + n_nn

    total_score = c_algorithm * n_total \
        + c_gp * (n_pp + n_pu + n_pn) \
        + c_specialist * (n_pu + n_up + n_uu + n_un) \
        + c_treatment * (n_pp + alpha * n_pu + n_up + alpha * n_uu) \
        + c_error * (n_np + alpha * n_nu)
    if n_total > 0:
        mean_score = total_score / n_total
    else:
        mean_score = float('nan')

    return mean_score

def get_pitch_label(data):
    label = None
    for l in data.split('\n'):
        if l.startswith('#Systolic murmur pitch:') or l.startswith('#Diastolic murmur pitch:'):
            try:
                label = l.split(': ')[1]
            except:
                pass
    if label is None:
        raise ValueError('No label available. Is your code trying to load labels from the hidden data?')
    return label