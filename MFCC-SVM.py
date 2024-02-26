from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
import librosa
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import label_binarize
from itertools import cycle


data = [
    ["MFCC-10"],
    ["MFCC-12"],
    ["MFCC-14"],
    ["MFCC-16"],
    ["MFCC-18"],
    ["MFCC-20"],
    ["MFCC-22"],
    ["MFCC-24"],
    ["MFCC-26"],
    ["MFCC-28"],
    ["MFCC-30"],
    ["MFCC-32"],
    ["MFCC-34"],
    ["MFCC-36"],
    ["MFCC-38"],
    ["MFCC-64"]
]
headers = ["Features", "SVM", "KNN", "NuSVC", "DecisionTreeClassifier", "MLPClassifier", "SGDClassifier", "LinearDiscriminantAnalysis", "RandomForestClassifier", "GaussianNB"]




data_dir = "C:/Users/MSI/Desktop/tiago1"


files = librosa.util.find_files(data_dir, recurse= True)


def calculate_mfcc(file, max_length , n_mfcc1):

    audio, sr = librosa.load(file)


    if len(audio) < max_length:
        audio = np.pad(audio, (0, max_length - len(audio)))
    else:
        audio = audio[:max_length]


    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc1, hop_length=100, n_fft=2048)

    return mfcc
f_counts = list(range(10, 39, 2))
f_counts.append(64)
f_num = 0
for f_count in f_counts:

    mfccs = []
    labels = []

    for file in files:
        parts = file.split("\\")
        for part in parts:
            if ".wav" in part:
                label = part
                break
        label1 = label.split("_")[0]
        labels.append(label1)

        # Calculate MFCCs for the current file
        mfcc = calculate_mfcc(file, 15000, f_count)
        mfccs.append(mfcc)

    mfccs = np.array(mfccs)
    labels = np.array(labels)

    mfccs = (mfccs - np.min(mfccs)) / (np.max(mfccs) - np.min(mfccs))


    mfccs_flat = np.reshape(mfccs, (mfccs.shape[0], -1))

    models = []
    models.append(make_pipeline(StandardScaler(), SVC(kernel='linear', probability= True)))
    models.append(make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)))
    models.append(make_pipeline(StandardScaler(), NuSVC(probability= True)))
    models.append(make_pipeline(StandardScaler(), DecisionTreeClassifier()))
    models.append(make_pipeline(StandardScaler(), MLPClassifier()))
    models.append(make_pipeline(StandardScaler(), SGDClassifier(loss ="modified_huber")))
    models.append(make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()))
    models.append(make_pipeline(StandardScaler(), RandomForestClassifier(n_jobs= -1 )))
    models.append(make_pipeline(StandardScaler(), GaussianNB()))

    for model in models:
        cv_accuracy = cross_val_score(model, mfccs_flat, labels, cv=10)
        data[f_num].append(np.mean(cv_accuracy))
        print(
            f'10-fold cross-validation accuracy: {np.mean(cv_accuracy):.2f} +/- {np.std(cv_accuracy):.2f} for MFCC-{f_count}')


print(tabulate(data, headers, tablefmt="grid"))

