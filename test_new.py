import numpy as np
from sklearn import preprocessing
import numpy.matlib

SD = np.loadtxt("SD.txt", dtype=float, delimiter=" ")
SM = np.loadtxt("SM.txt", dtype=float, delimiter=" ")
A = np.loadtxt("interaction.txt", dtype=int, delimiter=" ")
interacation = np.transpose(A)

R_B = np.matlib.repmat(interacation, 495, 1)
sm = np.repeat(SM, repeats=383, axis=0)
train_m = np.concatenate((sm, R_B), axis=1)  # (189585,990)

R_A = np.repeat(A, repeats=383, axis=0)
sd = np.matlib.repmat(SD, 495, 1)
train_s = np.concatenate((R_A, sd), axis=1)  # (189585,766)
label = A.reshape((189585, 1))


import tensorflow as tf
from tensorflow.keras import layers, models

encoding_dim = 64
input_mdata = layers.Input(shape=(990,))

encoded = layers.Dense(350, activation='relu')(input_mdata)
encoded = layers.Dense(250, activation='relu')(encoded)
encoded = layers.Dense(100, activation='relu')(encoded)
miRNA_encoder_output = layers.Dense(encoding_dim)(encoded)

decoded = layers.Dense(100, activation='relu')(miRNA_encoder_output)
decoded = layers.Dense(250, activation='relu')(decoded)
decoded = layers.Dense(350, activation='relu')(decoded)
decoded = layers.Dense(990, activation='sigmoid')(decoded)  # or 'tanh' if data is scaled accordingly

autoencoder = models.Model(input_mdata, decoded)
encoder = models.Model(input_mdata, miRNA_encoder_output)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(train_m, train_m, epochs=20, batch_size=100, shuffle=True)
miRNA_encoded_datas = encoder.predict(train_m)



encoding_dim = 64
input_ddata = layers.Input(shape=(766,))

encoded = layers.Dense(350, activation='relu')(input_ddata)
encoded = layers.Dense(250, activation='relu')(encoded)
encoded = layers.Dense(100, activation='relu')(encoded)
disease_encoder_output = layers.Dense(encoding_dim)(encoded)

decoded = layers.Dense(100, activation='relu')(disease_encoder_output)
decoded = layers.Dense(250, activation='relu')(decoded)
decoded = layers.Dense(350, activation='relu')(decoded)
decoded = layers.Dense(766, activation='sigmoid')(decoded)

autoencoder = models.Model(input_ddata, decoded)
encoder = models.Model(input_ddata, disease_encoder_output)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(train_s, train_s, epochs=20, batch_size=100, shuffle=True)
disease_encoded_datas = encoder.predict(train_s)


import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

X = np.hstack((miRNA_encoded_datas, disease_encoded_datas))
y = A.flatten()

num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True)

accuracy_results = []
precision_results = []
recall_results = []
f1_results = []
y_true = []
y_scores = []


#balanced in randomforestclassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Initialize lists to store evaluation metrics
accuracy_results, precision_results, recall_results, f1_results = [], [], [], []
y_true, y_scores = [], []

# KFold cross-validation
for train_index, val_index in kf.split(X):
    x_train, x_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Add class weights to the classifier
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced")
    clf.fit(x_train, y_train)

    # Get predictions and probabilities
    y_pred = clf.predict(x_val)
    y_pred_proba = clf.predict_proba(x_val)[:, 1]  # Probabilities for the positive class

    # Store true labels and predicted probabilities for ROC calculation
    y_true.extend(y_val)
    y_scores.extend(y_pred_proba)

    # Calculate metrics at default threshold (0.5)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    # Append metrics to the results lists
    accuracy_results.append(accuracy)
    precision_results.append(precision)
    recall_results.append(recall)
    f1_results.append(f1)

# Calculate average metrics across all folds
avg_accuracy = sum(accuracy_results) / len(accuracy_results)
avg_precision = sum(precision_results) / len(precision_results)
avg_recall = sum(recall_results) / len(recall_results)
avg_f1 = sum(f1_results) / len(f1_results)
roc_auc = roc_auc_score(y_true, y_scores)  # Overall ROC-AUC score

print(f"Average Accuracy: {avg_accuracy:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average F1 Score: {avg_f1:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
