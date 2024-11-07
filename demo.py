# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model, Sequential
from keras.utils import np_utils
import scipy.io

# Load data
print("Loading data...")
disease_features = np.loadtxt("integrated disease similarity.txt", dtype=float, delimiter="\t")
mirna_features = np.loadtxt("integrated circRNA similarity.txt", dtype=float, delimiter="\t")
interaction_matrix = np.loadtxt("Association Matrix.txt", dtype=int, delimiter="\t")

# Prepare training and testing data
num_associations = 0
training_data = []  
testing_data = []       
labels = []

for i in range(interaction_matrix.shape[0]):   
    for j in range(interaction_matrix.shape[1]): 
        if interaction_matrix[i, j] == 1:  # associated
            labels.append(1)  # label for association (1)
            num_associations += 1  # count of associated samples
            training_data.append((mirna_features[i].tolist(), disease_features[j].tolist()))  # feature vector for association
        elif interaction_matrix[i, j] == 0:  # no association
            labels.append(0)  # label for no association (0)
            testing_data.append((mirna_features[i].tolist(), disease_features[j].tolist()))  # feature vector for no association

print("Number of associations:", num_associations)

# Randomly shuffle the non-associated samples
shuffle_indices = np.arange(len(labels) - num_associations)
np.random.shuffle(shuffle_indices)
shuffled_testing_data = [testing_data[index] for index in shuffle_indices[:num_associations]]
shuffled_labels = [labels[index] for index in shuffle_indices[:num_associations]]

# Equalize positive and negative samples for training
for index in range(num_associations):  # append non-associated feature vectors to training data
    training_data.append(shuffled_testing_data[index])  
    labels.append(shuffled_labels[index])  # append the labels of non-associated pairs (0) to labels

# Convert training data to numpy array
training_data = np.array(training_data)

# Split features into separate arrays
mirna_features = np.array([item[0] for item in training_data])
disease_features = np.array([item[1] for item in training_data])

# Combine the features
combined_features = np.concatenate((mirna_features, disease_features), axis=1)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels).astype(np.int32)
categorical_labels = np_utils.to_categorical(encoded_labels)

# Build and train the autoencoder
autoencoder_model = Sequential()
autoencoder_model.add(Dense(500, input_dim=673, activation='relu'))
autoencoder_model.add(Dropout(0.3))
autoencoder_model.add(Dense(500, activation='relu'))
autoencoder_model.add(Dropout(0.3))
autoencoder_model.add(Dense(300, activation='relu'))
autoencoder_model.add(Dropout(0.3))
autoencoder_model.add(Dense(2, activation='sigmoid'))

autoencoder_model.compile(loss='categorical_crossentropy', optimizer='Adadelta')

# Train the autoencoder
autoencoder_model.fit(combined_features, combined_features, epochs=20, batch_size=100, shuffle=True)

# Predict encoded features
encoded_features = autoencoder_model.predict(combined_features)

# Cross-validation and performance evaluation
num_cross_validation_folds = 5
all_performance_metrics = []
mean_false_positive_rate = np.linspace(0, 1, 100)

for fold in range(num_cross_validation_folds):
    # Create training and testing datasets for this fold
    fold_training_data = np.array([encoded_features[i] for i in range(len(encoded_labels)) if i % num_cross_validation_folds != fold])
    fold_testing_data = np.array([encoded_features[i] for i in range(len(encoded_labels)) if i % num_cross_validation_folds == fold])
    fold_training_labels = np.array([categorical_labels[i] for i in range(len(encoded_labels)) if i % num_cross_validation_folds != fold])
    fold_testing_labels = np.array([categorical_labels[i] for i in range(len(encoded_labels)) if i % num_cross_validation_folds == fold])

    true_labels = [0 if val[0] == 1 else 1 for val in fold_testing_labels]
    training_labels_flat = [0 if val[0] == 1 else 1 for val in fold_training_labels]

    # Train the XGBoost classifier
    xgb_classifier = XGBClassifier(n_estimators=250, learning_rate=0.2, max_depth=15)
    xgb_classifier.fit(fold_training_data, training_labels_flat)

    # Make predictions
    predicted_probabilities = xgb_classifier.predict_proba(fold_testing_data)[:, 1]

    # Calculate performance metrics
    false_positive_rate, true_positive_rate, _ = roc_curve(true_labels, predicted_probabilities)
    auc_score = auc(false_positive_rate, true_positive_rate)

    # Calculate additional metrics
    tp = np.sum((true_labels == 1) & (predicted_probabilities >= 0.5))
    fp = np.sum((true_labels == 0) & (predicted_probabilities >= 0.5))
    tn = np.sum((true_labels == 0) & (predicted_probabilities < 0.5))
    fn = np.sum((true_labels == 1) & (predicted_probabilities < 0.5))

    accuracy = (tp + tn) / len(true_labels)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    sensitivity = tp / (tp + fn) if tp + fn > 0 else 0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0
    mcc = (tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 0
    f1_score = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    # Save results
    scipy.io.savemat('raw_DNN', {'fpr': false_positive_rate, 'tpr': true_positive_rate, 'auc_score': auc_score})
    
    print("Fold {} Performance: Accuracy = {:.4f}, Precision = {:.4f}, Sensitivity = {:.4f}, Specificity = {:.4f}, MCC = {:.4f}, AUC = {:.4f}, F1 Score = {:.4f}".format(fold + 1, accuracy, precision, sensitivity, specificity, mcc, auc_score, f1_score))
    all_performance_metrics.append([accuracy, precision, sensitivity, specificity, mcc, auc_score, f1_score])
    
    plt.plot(false_positive_rate, true_positive_rate, label='ROC fold %d (AUC = %0.4f)' % (fold + 1, auc_score))

# Average ROC Curve
mean_false_positive_rate = np.linspace(0, 1, 100)
mean_true_positive_rate = np.mean([metrics[1] for metrics in all_performance_metrics], axis=0)
mean_true_positive_rate[-1] = 1.0
mean_auc = auc(mean_false_positive_rate, mean_true_positive_rate)

plt.plot(mean_false_positive_rate, mean_true_positive_rate, '--', linewidth=2.5, label='Mean ROC (AUC = %0.4f)' % mean_auc)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.savefig('ROC_curve.png')
plt.show()
