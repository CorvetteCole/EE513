from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report, accuracy_score
import glob
import librosa
import logging

from quiz4.formant_frequencies import find_formant_frequencies

logging.getLogger('numba').setLevel(logging.WARNING)

sample_dir = Path('samples')


def extract_fundamental_frequency(audio_signal, sr):
    # extract pitch info
    pitches, magnitudes = librosa.core.piptrack(y=audio_signal, sr=sr)

    # Selecting the maximum magnitude within each time frame to determine the pitch
    # Zero magnitude frames are ignored
    index_of_max_magnitudes = magnitudes.argmax(axis=0)
    pitch_track = pitches[index_of_max_magnitudes, np.arange(pitches.shape[1])]

    # Filtering out zero pitch values (unvoiced segments)
    pitch_track = pitch_track[pitch_track > 0]

    if len(pitch_track) == 0:
        # In case there is no voiced segment, return 0 or any representative value
        f0 = 0
    else:
        f0 = np.median(pitch_track)

    return f0


def main():
    # Configuration
    n_splits = 5
    sampling_rate = 16000
    n_features = 3  # Recommended to limit to 12 features

    # Load the audio samples and preprocess them if necessary
    filepaths_yes = glob.glob(f'{sample_dir}/y*.wav')
    filepaths_no = glob.glob(f'{sample_dir}/n*.wav')

    # Extract features for each audio file
    features = []

    for filepath in filepaths_yes + filepaths_no:
        signal, _ = librosa.load(filepath, sr=sampling_rate)
        # signal, _ = librosa.effects.trim(signal, top_db=20)
        signal = librosa.util.normalize(signal)

        # Extract formant frequencies
        lpc_coefficients = librosa.lpc(signal, order=2 * n_features)
        formant_features = find_formant_frequencies(lpc_coefficients)[:n_features]

        if len(formant_features) < n_features:
            formant_features = np.pad(formant_features, (0, n_features - len(formant_features)), 'constant')

        # get fundamental frequency
        f0 = extract_fundamental_frequency(signal, sampling_rate)

        feature_vector = np.concatenate((formant_features, [f0]))

        features.append(feature_vector)

    x = np.array(features)
    y = np.array([1] * len(filepaths_yes) + [0] * len(filepaths_no))  # Label "yes" as 1 and "no" as 0

    kf = KFold(n_splits=n_splits, shuffle=True)

    # Initialize arrays to track performance metrics
    conf_matrix = np.zeros((2, 2))  # Binary classification -> 2x2 matrix

    # Initialize lists to track true labels and predicted labels during cross-validation
    true_labels = []
    predicted_labels = []
    accuracies = []

    # Perform K-fold cross-validation
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Create and train the model
        clf = GaussianNB()
        clf.fit(x_train, y_train)

        # Make predictions on the test fold
        y_pred = clf.predict(x_test)

        # Update lists of true labels and predictions
        true_labels.extend(y_test)
        predicted_labels.extend(y_pred)

        # Update confusion matrix
        conf_matrix += confusion_matrix(y_test, y_pred)

        # Compute and store the accuracy for the current fold
        fold_accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(fold_accuracy)

    std_dev_accuracy = np.std(accuracies)

    # Now that we have predictions from each fold, compute overall performance metrics
    precisions, recalls, f1_scores, _ = precision_recall_fscore_support(true_labels, predicted_labels, average=None)

    precision_yes = precisions[1]
    recall_yes = recalls[1]
    precision_no = precisions[0]
    recall_no = recalls[0]

    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=['no', 'yes']))

    print("Performance:")
    print("Recall for 'Yes': {:.2f}%".format(100 * recall_yes))
    print("Precision for 'Yes': {:.2f}%".format(100 * precision_yes))
    print("Recall for 'No': {:.2f}%".format(100 * recall_no))
    print("Precision for 'No': {:.2f}%".format(100 * precision_no))
    print("Standard Deviation of Accuracy over Cross-Validation: {:.4f}".format(std_dev_accuracy))

    conf_matrix_formatted = np.array([
        [conf_matrix[0, 0], conf_matrix[0, 1], 100 * precision_no],
        [conf_matrix[1, 0], conf_matrix[1, 1], 100 * precision_yes],
        [100 * recall_no, 100 * recall_yes, np.nan]
    ])

    print("\nRequired Confusion Matrix Format:")
    print(conf_matrix_formatted)


if __name__ == '__main__':
    main()
