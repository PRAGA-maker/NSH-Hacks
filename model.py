import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
import librosa
import random
import os
import tqdm
import tensorflow as tf
from tensorflow.keras import backend as K
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from keras.optimizers import SGD
import keras
from sklearn.utils.class_weight import compute_class_weight
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout,Masking


import george_moody_challenge_2022_utility_script as gmc
np.random.seed(42)
keras.backend.clear_session()
random.seed(42)




def get_filepaths_and_labels():
    data_path= r"C:\Users\prone\OneDrive\Desktop\NSH Hackathon\mod_data"
    patient_files = gmc.find_patient_files(data_path)
    num_patient_files = len(patient_files)
    print("Num patient files: " + str(num_patient_files))
    classes = ['Present', 'Absent']


    labels = []
    filepaths = []
    benigns = []
    for i in tqdm.tqdm(range(num_patient_files-100)):
            patient_id = ((patient_files[i]))[-9:-4]
            patient_txt = patient_files[i]
            current_patient_data = gmc.load_patient_data(patient_files[i])
            pregnant = gmc.get_pregnancy_status(current_patient_data)
            if pregnant == False:
                label = gmc.get_label(current_patient_data)
                if label in classes:
                    for root, dirs, files in os.walk(data_path):
                        for filename in files:
                            if filename.startswith(patient_id) and (filename.endswith("txt") == False) and (filename.endswith("hea") == False) and (filename.endswith("tsv") == False): #and ("PV" in filename)
                                filepaths.append(os.path.join(root, filename))
                                labels.append(label)
                                if label == "Present":
                                    try:
                                        with open(patient_txt, 'r') as file:
                                            lines = file.readlines()
                                            for line in lines:
                                                if line.startswith("#Outcome:"):
                                                    line = line.split(" ")
                                                    benign = line[1]
                                                    if "Abnormal" in benign:
                                                        benigns.append(True)
                                                    else:
                                                        benigns.append(False)
                                    except UnicodeDecodeError:
                                        print("UnicodeDecodeError")
   
    num_benign = 0
    num_abnormal = 0
    for index in benigns:
        if index == False:
            num_benign+=1
        elif index == True:
            num_abnormal +=1
    print("Num Benign Murmurs: " + str(num_benign))
    print("Num Abnormal Murmurs: " + str(num_abnormal))


    num_present = 0
    num_absent = 0
    for label in labels:
        if label == "Present":
            num_present+=1
        elif label == "Absent":
            num_absent +=1
    print("Num Present: " + str(num_present))
    print("Num Absent: " + str(num_absent))


    return filepaths,labels


# filepaths of the audio files and corresponding labels ("Present" or "Absent")
filepaths, labels = get_filepaths_and_labels()


# max_absent = 561
# new_labels = []
# new_filepaths = []
# for i in range(len(labels)):
#     if (len(new_labels) > max_absent) == False:
#         if labels[i] == "Absent":
#             new_labels.append(labels[i])
#             new_filepaths.append(filepaths[i])
#     if labels[i] == "Present":
#         new_labels.append(labels[i])
#         new_filepaths.append(filepaths[i])
# filepaths = new_filepaths
# labels = new_labels


#  labels from strings to integers (0 or 1)
labels = [0 if label == "Absent" else 1 for label in labels]


def read_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr


def resample_audio(y, sr, target_sr=100):
    y_resampled = librosa.resample(y, sr, target_sr)
    return y_resampled, target_sr


def apply_compression(y, threshold=-20.0, ratio=4.0):
    gain = lambda x: 1.0 / ratio if x < threshold else 1.0
    return np.array([sample * gain(sample) for sample in y])


def apply_distortion(y, amount):
    return np.clip(y + amount * y**3, -1, 1)


def apply_gating(y, threshold=0.01):
    return np.where(np.abs(y) > threshold, y, 0.0)


def calculate_rmse(original, processed):
    diff = original[:len(processed)] - processed
    rmse = np.sqrt(np.mean(diff**2))
    return rmse


def apply_packet_loss(y, sr, packet_loss_rate, packet_duration=0.02):
    packet_length = int(packet_duration * sr)
    num_packets = len(y) // packet_length
    packets_to_drop = int(num_packets * packet_loss_rate)


    drop_indices = np.random.choice(num_packets, packets_to_drop, replace=False)
    y_dropped = []

    for i in range(num_packets):
        if i not in drop_indices:
            y_dropped.extend(y[i*packet_length : (i+1)*packet_length])


    return np.array(y_dropped)


def apply_packet_delay(y, sr, delay_duration):
    delay_samples = int(delay_duration * sr)
    silence = np.zeros(delay_samples)
    return np.concatenate((silence, y))


# Augmentation function
def augment_audio(y, sr):
    y_resampled, sr_resampled = resample_audio(y, sr)
    y_compressed = apply_compression(y_resampled)
    y_distorted = apply_distortion(y_compressed,amount=0.1)
    y_gated = apply_gating(y_distorted)
    y = y_gated
    roll = random.randint(1, 3)
    if roll == 1:
        ms_roll = (random.randint(1,100))/1000
        y = apply_packet_delay(y, sr, delay_duration=ms_roll)
    elif roll == 2:
        percent_roll = (random.randint(1,30))/100
        y = apply_packet_loss(y, sr, packet_loss_rate=percent_roll)
    elif roll == 3:
        percent_roll = (random.randint(1,30))/100
        y = apply_distortion(y, amount=percent_roll)
    return y


def create_distribution_chart(data, title="Distribution Chart", xlabel="Values", ylabel="Frequency"):
    plt.hist(data, bins=20, edgecolor='k')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def ensure_consistent_shape(features, max_length=19620):  #64*7 32256
    if features.size < max_length:
        padding = np.zeros(max_length - features.size)
        features = np.hstack([features, padding])
    elif features.size > max_length:
        #print(features.size)
        #print(features[max_length:])
        features = features[:max_length]
    return features


# Feature extraction function
def extract_features(file_path):
    y, sr = read_audio(file_path)
   
    # Augment audio
    y = augment_audio(y,sr)
    #spectral_contrasts = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=4, n_fft=2048)
    #spectral_bandwidths = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=2048)
    #zero_crossing_rates = librosa.feature.zero_crossing_rate(y)


    # features = np.vstack([spectral_contrasts, spectral_bandwidths, zero_crossing_rates])
    # features = features.flatten()
    features = y
    features = np.array(features)
    features = ensure_consistent_shape(features)
    return features


# Extract features from all audio files
X = np.vstack([extract_features(fp) for fp in filepaths])


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)


# Convert labels to one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


input_shape = (X_train.shape[1],)
n_splits = 5


# Initialize the StratifiedKFold object
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#@keras.utils.register_keras_serializable()
def weighted_binary_crossentropy(y_true, y_pred):
    # Define your class weights here (e.g., inversely proportional to class frequencies)
    class_weights = tf.constant([1.0, 2.0])  # Adjust these values as needed


    # Calculate weighted binary cross-entropy
    weighted_losses = class_weights * (y_true * K.log(y_pred + K.epsilon()) + (1 - y_true) * K.log(1 - y_pred + K.epsilon()))
    return -K.mean(weighted_losses)


# Create lists to store your cross-validation results
cv_scores = []


# Convert labels to one-hot encoding
labels_one_hot = to_categorical(labels)


# Initialize variables for visualizations
train_accuracy_history = []
val_accuracy_history = []


# Iterate over the folds
for train_index, test_index in skf.split(X, labels):
    X_train, X_val = X[train_index], X[test_index]
    y_train, y_val = labels_one_hot[train_index], labels_one_hot[test_index]


    # Create a new model for each fold to ensure independence
    model = Sequential()
    #model.add(Masking(mask_value=0.0,input_shape=(X_train.shape[1],1)))
    model.add(Dense(612, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))


    optimizer1 = SGD(learning_rate=0.01, momentum=0.9)
    class_weights = compute_class_weight('balanced', classes=[0, 1], y=labels)
    model.compile(optimizer=optimizer1, loss=weighted_binary_crossentropy, metrics=['accuracy'])


    # Train the model on the current fold
    history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_val, y_val))
   
    # Append training and validation accuracy to history
    train_accuracy_history.append(history.history['accuracy'])
    val_accuracy_history.append(history.history['val_accuracy'])


    # Evaluate the model on the validation set
    scores = model.evaluate(X_val, y_val)
    cv_scores.append(scores)


# Calculate the mean and standard deviation of the cross-validation scores
mean_score = np.mean(cv_scores)
std_score = np.std(cv_scores)


# Print the results
print(f"Mean CV Score: {mean_score}, Standard Deviation: {std_score}")


# Visualization
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


plt.figure(figsize=(15, 5))


# Plot the training and validation accuracy
for i in range(n_splits):
    plt.plot(train_accuracy_history[i], label=f'Fold {i + 1} Train Accuracy', linestyle='--')
    plt.plot(val_accuracy_history[i], label=f'Fold {i + 1} Validation Accuracy')


plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# AUC-ROC curve
y_pred = model.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_pred[:, 1])
roc_auc = auc(fpr, tpr)


plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)


# Calculate the confusion matrix
confusion = confusion_matrix(y_true, y_pred_classes)


# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion, display_labels=["Present", "Absent"])
disp.plot(cmap='viridis')
plt.title('Confusion Matrix')
plt.show()


# Save the model
model.save('my_model.keras')




def preprocess_audio(file_path, delay=0, loss=0, distortion=0):
    y, sr = read_audio(file_path)
    y_resampled, sr_resampled = resample_audio(y, sr)
    y_compressed = apply_compression(y_resampled)
    y_distorted = apply_distortion(y_compressed,amount=0.2)
    y_gated = apply_gating(y_distorted)
    y = y_gated
   
    # Apply delay
    y = apply_packet_delay(y, sr, delay_duration=delay/1000)  # Convert ms to s
   
    # Apply packet loss
    y = apply_packet_loss(y, sr, packet_loss_rate=loss/100)
   
    # Apply distortion
    y = apply_distortion(y, distortion/100)
   
    # spectral_contrasts = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=4, n_fft=2048)
    # spectral_bandwidths = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=2048)
    # zero_crossing_rates = librosa.feature.zero_crossing_rate(y)
    # features = np.vstack([spectral_contrasts, spectral_bandwidths, zero_crossing_rates])
    # features = features.flatten()
    features = y
    features = np.array(features)
    features = ensure_consistent_shape(features)
    return features


conditions = [
    (0,0,0),
    (33,0,0),
    (66,0,0),
    (99,0,0),
    (0,10,0),
    (0,20,0),
    (0,30,0),
    (0,0,10),
    (0,0,20),
    (0,0,30),
    (99,30,30)
]


results_matrix = []


for condition in conditions:
    delay, loss, distortion = condition
    data_path= r"C:\Users\prone\auscultation\data\training_data"
    patient_files = gmc.find_patient_files(data_path)
    num_patient_files = len(patient_files)
    classes = ['Present', 'Absent']
    labels = []
    filepaths = []
    for i in tqdm.tqdm(range(num_patient_files-100,num_patient_files)):
            patient_id = ((patient_files[i]))[-9:-4]
            patient_txt = patient_files[i]
            current_patient_data = gmc.load_patient_data(patient_files[i])
            label = gmc.get_label(current_patient_data)
            if label in classes:
                for root, dirs, files in os.walk(data_path):
                    for filename in files:
                        if filename.startswith(patient_id) and (filename.endswith("txt") == False) and (filename.endswith("hea") == False) and (filename.endswith("tsv") == False):
                            filepaths.append(os.path.join(root, filename))
                            labels.append(label)


    X_test_processed = np.array([preprocess_audio(fp, delay, loss, distortion) for fp in filepaths])
   
    # Convert labels from strings to integers (0 or 1)
    labels = [0 if label == "Absent" else 1 for label in labels]
    y_test = labels
    y_test = to_categorical(y_test)
    _, accuracy = model.evaluate(X_test_processed, y_test, verbose=0)
   
    y_pred = model.predict(X_test_processed)
    precision = np.sum((y_pred[:, 1] > 0.5) & (y_test[:, 1] == 1)) / np.sum(y_pred[:, 1] > 0.5)
   
    results_matrix.append((delay, loss, distortion, accuracy, precision))


# Convert to a numpy array for better visualization
results_matrix = np.array(results_matrix)


df = pd.DataFrame(results_matrix, columns=['Delay', 'Loss', 'Distortion', 'Accuracy', 'Precision'])
print(df)


# You can also visualize it using seaborn's heatmap for a more visual representation.
plt.figure(figsize=(12, 6))
sns.heatmap(df.drop(['Delay', 'Loss', 'Distortion'], axis=1), annot=True, cmap='viridis')
plt.show()



