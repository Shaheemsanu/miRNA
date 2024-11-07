# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model, Sequential
from keras.utils import np_utils
import scipy.io

def prepare_data():
    print("Loading data...")
    
    disease_fea = np.loadtxt("integrated disease similarity.txt", dtype=float, delimiter="\t")
    miRNA_fea = np.loadtxt("integrated circRNA similarity.txt", dtype=float, delimiter="\t")
    interaction = np.loadtxt("Association Matrix.txt", dtype=int, delimiter="\t")

    link_number = 0
    train = []  
    testfnl = []       
    label1 = []
    label2 = []
    
    for i in range(interaction.shape[0]):   
        for j in range(interaction.shape[1]): 
            if interaction[i, j] == 1:  # for associated
                label1.append(interaction[i, j])  # labels for association (1)
                link_number += 1  # number of associated samples
                train.append((miRNA_fea[i].tolist(), disease_fea[j].tolist()))  # feature vector for association
            elif interaction[i, j] == 0:  # for no association
                label2.append(interaction[i, j])  # labels for no association (0)
                testfnl.append((miRNA_fea[i].tolist(), disease_fea[j].tolist()))  # feature vector for no association

    print("Link number:", link_number)
    
    # Randomly shuffle the negative samples
    m = np.arange(len(label2))
    np.random.shuffle(m)
    
    ttfnl = [testfnl[x] for x in m]
    label22 = [label2[x] for x in m]
    
    for x in range(link_number):  # for equalizing positive and negative samples
        train.append(ttfnl[x])  # append the non-associated feature vector pairs to train
        label1.append(label22[x])  # append the labels of non-associated pairs (0) to label1

    return np.array(train), label1, np.array(testfnl)

def calculate_performance(test_num, pred_y, labels):
    tp, fp, tn, fn = 0, 0, 0, 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp += 1
            else:
                fn += 1
        else:
            if labels[index] == pred_y[index]:
                tn += 1
            else:
                fp += 1               
            
    acc = float(tp + tn) / test_num
    precision = float(tp) / (tp + fp) if tp + fp > 0 else 0
    sensitivity = float(tp) / (tp + fn) if tp + fn > 0 else 0
    specificity = float(tn) / (tn + fp) if tn + fp > 0 else 0
    MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 0
    f1_score = float(2 * tp) / ((2 * tp) + fp + fn) if (2 * tp) + fp + fn > 0 else 0

    return acc, precision, sensitivity, specificity, MCC, f1_score

def transfer_array_format(data):
    formated_matrix1 = []
    formated_matrix2 = []
    for val in data:
        formated_matrix1.append(val[0])  # contains miRNA features
        formated_matrix2.append(val[1])  # contains disease features
    
    return np.array(formated_matrix1), np.array(formated_matrix2)

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def DNN():
    model = Sequential()
    model.add(Dense(500, input_dim=128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='sigmoid'))
    
    model.compile(loss='categorical_crossentropy', optimizer='Adadelta')
    return model

def DNN_auto(x_train):
    input_img = Input(shape=(673,))
    encoded = Dense(500, activation='relu')(input_img)
    encoded = Dense(128, activation='relu')(encoded)
    encoder_output = Dense(128)(encoded)

    decoded = Dense(128, activation='relu')(encoder_output)
    decoded = Dense(500, activation='relu')(decoded)
    decoded = Dense(673, activation='tanh')(decoded)

    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoder_output)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(x_train, x_train, epochs=20, batch_size=100, shuffle=True)
    
    encoded_imgs = encoder.predict(x_train)
    print("Encoded images shape:", encoded_imgs.shape)  # Output shape of encoded images
    return encoder_output, encoded_imgs

def DeepMDA():
    X, labels, T = prepare_data()  # Load data
    X_data1, X_data2 = transfer_array_format(X)
    
    print("Shapes:", X_data1.shape, X_data2.shape)
    
    X_data1 = np.concatenate((X_data1, X_data2), axis=1)
    
    print("Concatenated shape:", X_data1.shape)
    
    y, encoder = preprocess_labels(labels)
    num = np.arange(len(y))
    np.random.shuffle(num)
    X_data1 = X_data1[num]
    y = y[num]
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    
    encoder, X_data1 = DNN_auto(X_data1)  # Auto encode X_data1
    
    num_cross_val = 5
    all_performance_DNN = []
    
    for fold in range(num_cross_val):
        train1 = np.array([x for i, x in enumerate(X_data1) if i % num_cross_val != fold])
        test1 = np.array([x for i, x in enumerate(X_data1) if i % num_cross_val == fold])
        train_label = np.array([x for i, x in enumerate(y) if i % num_cross_val != fold])
        test_label = np.array([x for i, x in enumerate(y) if i % num_cross_val == fold])

        real_labels = [0 if val[0] == 1 else 1 for val in test_label]
        train_label_new = [0 if val[0] == 1 else 1 for val in train_label]
        
        clf = XGBClassifier(n_estimators=250, learning_rate=0.2, max_depth=15)
        clf.fit(train1, train_label_new)
        
        ae_y_pred_prob = clf.predict_proba(test1)[:, 1]
        
        proba = transfer_label_from_prob(ae_y_pred_prob)
        
        acc, precision, sensitivity, specificity, MCC, f1_score = calculate_performance(len(real_labels), proba, real_labels)
        
        fpr, tpr, _ = roc_curve(real_labels, ae_y_pred_prob)
        auc_score = auc(fpr, tpr)
        
        # Save results
        scipy.io.savemat('raw_DNN', {'fpr': fpr, 'tpr': tpr, 'auc_score': auc_score})
        
        print("AUTO-RF:", acc, precision, sensitivity, specificity, MCC, auc_score, f1_score)
        all_performance_DNN.append([acc, precision, sensitivity, specificity, MCC, auc_score, f1_score])
        
        plt.plot(fpr, tpr, label='ROC fold %d (AUC = %0.4f)' % (fold + 1, auc_score))

    mean_tpr /= num_cross_val
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    plt.plot(mean_fpr, mean_tpr, '--', linewidth=2.5, label='Mean ROC (AUC = %0.4f)' % mean_auc)
    plt.xlabel('False positive rate, (1-Specificity)')
    plt.ylabel('True positive rate, Sensitivity')
    plt.title('ROC')
    plt.legend(loc='best')
    plt.savefig('ROC_curve.png')
    plt.show()

def main():
    DeepMDA()

if __name__ == "__main__":
    main()
