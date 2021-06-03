"""
Description: Utility functions for evaluations.
"""

import os
from os.path import dirname, abspath

import torch as th
import torch.utils.data

from utils.utils import tsne
import csv
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from collections import Counter


def linear_model_eval(config, X_train, y_train, X_test, y_test, use_scaler=False, description="Logistic Reg."):
    """
    :param ndarray X_train:
    :param list y_train:
    :param ndarray X_test:
    :param list y_test:
    :param bool use_scaler:
    :param str description:
    :return:
    """
    results_list = []
    
    c=1e7
    
    for c in [1, 10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]:
        # Initialize Logistic regression
        print(10*"*"+"C="+str(c)+10*"*")
        clf = LogisticRegression(max_iter=1200, solver='lbfgs', C=c, multi_class='multinomial') #RandomForestClassifier(n_estimators=100)  #
        # Fit model to the data
        clf.fit(X_train, y_train)
        # Summary of performance
        print(10 * ">" + description)
        tr_acc = clf.score(X_train, y_train)
        te_acc = clf.score(X_test, y_test)
        print("Train score:", tr_acc)
        print("Test score:", te_acc)

        results_list.append({"model":"LogReg_"+str(c), "train_acc":tr_acc, "train_std":0, "test_acc":te_acc, "test_std":0})
    
    file_name = "SubTab_"+str(config["n_subsets"])+"_o_"+str(config["overlap"])+"_b_"+str(config["batch_size"])
    file_name = file_name + "_d_"+str(config["dims"][-1])+"_e_"+str(config["epochs"])+"_s_"+str(config["seed"])
    
    keys = results_list[0].keys()
    with open('./results/'+file_name+'.csv', 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results_list)
    

def plot_clusters(config, z, clabels, plot_suffix="_inLatentSpace"):
    # Number of columns for legends, where each column corresponds to a cluster
    ncol = len(list(set(clabels)))
    # clegends = ["A", "B", "C", "D", ...]..choose first ncol characters, one per cluster
    clegends = list("0123456789")[0:ncol]

    # Show clusters only
    visualise_clusters(config, z, clabels, plt_name="classes" + plot_suffix, legend_title="Classes", legend_labels=clegends)


def visualise_clusters(config, embeddings, labels, plt_name="test", alpha=1.0, legend_title=None, legend_labels=None,
                       ncol=1):
    """
    :param ndarray embeddings: Latent representations of samples.
    :param ndarray labels: Class labels;
    :param plt_name: Name to be used when saving the plot.
    :return: None
    """
    # Define colors to be used for each class/cluster
    color_list = ['#66BAFF', '#FFB56B', '#8BDD89', '#faa5f3', '#fa7f7f',
                  '#008cff', '#ff8000', '#04b000', '#de4bd2', '#fc3838',
                  '#004c8b', "#964b00", "#026b00", "#ad17a1", '#a80707',
                  "#00325c", "#e41a1c", "#008DF9", "#570950", '#732929']

    color_list2 = ['#66BAFF', '#008cff', '#004c8b', '#00325c',
                   '#FFB56B', '#ff8000', '#964b00', '#e41a1c',
                   '#8BDD89', "#04b000", "#026b00", "#008DF9",
                   "#faa5f3", "#de4bd2", "#ad17a1", "#570950",
                   '#fa7f7f', '#fc3838', '#a80707', '#732929']

    # If there are more than 3 types of labels, change color scheme. -- Not Used
    color_list = color_list2 if len(list(set(labels))) > config["n_classes"] + 1 else color_list

    # Map class to legend texts. -- Not Used
    c2l = {"0": "A1", "1": "A2", "2": "A3", "3": "A4",
           "4": "B1", "5": "B2", "6": "B3", "7": "B4",
           "8": "C1", "9": "C2", "10": "C3", "11": "C4",
           "12": "D1", "13": "D2", "14": "D3", "15": "D4",
           "16": "E1", "17": "E2", "18": "E3", "19": "E4", }

    # Used to adjust space for legends based on number of columns in the legend. ncol: subplot_adjust
    legend_space_adjustment = {"1": 0.9, "2": 0.9, "3": 0.75, "4": 0.65, "5": 0.65}

    # Initialize an empty dictionary to hold the mapping for color palette
    palette = {}
    # Map colors to the indexes.
    for i in range(len(color_list)):
        palette[str(i)] = color_list[i]
    # Make sure that the labels are 1D arrays
    y = labels.reshape(-1, )
    # Turn labels to a list
    y = list(map(str, y.tolist()))
    # Define number of sub-plots to draw. In this case, 2, one for PCA, and one for t-SNE
    img_n = 2
    # Initialize subplots
    fig, axs = plt.subplots(1, img_n, figsize=(9, 3.5), facecolor='w', edgecolor='k')
    # Adjust the whitespace around sub-plots
    fig.subplots_adjust(hspace=.1, wspace=.1)
    # adjust the ticks of axis.
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',
        left=False,  # both major and minor ticks are affected
        right=False,
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)

    # Flatten axes if we have more than 1 plot. Or, return a list of 2 axs to make it compatible with multi-plot case.
    axs = axs.ravel() if img_n > 1 else [axs, axs]

    # Get 2D embeddings, using PCA
    pca = PCA(n_components=2)
    # Fit training data and transform
    embeddings_pca = pca.fit_transform(embeddings)  # if embeddings.shape[1]>2 else embeddings
    # Set the title of the sub-plot
    axs[0].title.set_text('Embeddings from PCA')
    # Plot samples, using each class label to define the color of the class.
    sns_plt = sns.scatterplot(x=embeddings_pca[:, 0], y=embeddings_pca[:, 1], ax=axs[0], palette=palette, hue=y, s=20,
                              alpha=alpha)
    # Overwrite legend labels
    overwrite_legends(sns_plt, c2l, fig, ncol=ncol, title=legend_title, labels=legend_labels)
    # Get 2D embeddings, using t-SNE
    embeddings_tsne = tsne(embeddings)  # if embeddings.shape[1]>2 else embeddings
    # Set the title of the sub-plot
    axs[1].title.set_text('Embeddings from t-SNE')
    # Plot samples, using each class label to define the color of the class.
    sns_plt = sns.scatterplot(x=embeddings_tsne[:, 0], y=embeddings_tsne[:, 1], ax=axs[1], palette=palette, hue=y, s=20,
                              alpha=alpha)
    # Overwrite legend labels
    overwrite_legends(sns_plt, c2l, fig, ncol=ncol, title=legend_title, labels=legend_labels)
    # Remove legends in sub-plots
    axs[0].get_legend().remove()
    axs[1].get_legend().remove()
    # Adjust the scaling factor to fit your legend text completely outside the plot
    # (smaller value results in more space being made for the legend)
    plt.subplots_adjust(right=legend_space_adjustment[str(ncol)])

    # Get the path to the project root
    root_path = os.path.dirname(os.path.dirname(__file__))
    # Define the path to save the plot to.
    fig_path = os.path.join(root_path, "results", config["framework"], "evaluation", "clusters", plt_name + ".png")
    # Define tick params
    plt.tick_params(axis=u'both', which=u'both', length=0)
    # Save the plot
    plt.savefig(fig_path, bbox_inches="tight")
    # Clear figure just in case if there is a follow-up plot.
    plt.clf()


def overwrite_legends(sns_plt, c2l, fig, ncol, title=None, labels=None):
    # Get legend handles and labels
    handles, legend_txts = sns_plt.get_legend_handles_labels()
    # Turn str to int before sorting ( to avoid wrong sort order such as having '10' in front of '4' )
    legend_txts = [int(d) for d in legend_txts]
    # Sort both handle and texts so that they show up in a alphabetical order on the plot
    legend_txts, handles = (list(t) for t in zip(*sorted(zip(legend_txts, handles))))
    # Turn int to str before using labels
    legend_txts = [str(i) for i in legend_txts]
    # Get new legend labels using class-to-label map
    new_labels = [c2l[legend_text] for legend_text in legend_txts]
    # Overwrite new_labels if it is given by user.
    new_labels = labels or new_labels
    # Define the figure title
    title = title or "Cluster"
    # Overwrite the legend labels and add a title to the legend
    fig.legend(handles, new_labels, loc="center right", borderaxespad=0.1, title=title, ncol=ncol)
    sns_plt.set(xticklabels=[], yticklabels=[], xlabel=None, ylabel=None)
    sns_plt.tick_params(top=False, bottom=False, left=False, right=False)


def save_np2csv(np_list, save_as="test.csv"):
    # Get numpy arrays and label lists
    Xtr, ytr = np_list
    # Turn label lists into numpy arrays
    ytr = np.array(ytr, dtype=np.int8)
    # Get column names
    columns = ["label"] + list(map(str, list(range(Xtr.shape[1]))))

    # Concatenate "scaled" features and labels
    data_tr = np.concatenate((ytr.reshape(-1, 1), Xtr), axis=1)
    # Generate new dataframes with "scaled features" and labels
    df_tr = pd.DataFrame(data=data_tr, columns=columns)
    # Show samples from scaled data
    print("Samples from the dataframe:")
    print(df_tr.head())
    # Save the dataframe as csv file
    df_tr.to_csv(save_as, index=False)
    # Print an informative message
    print(f"The dataframe is saved as {save_as}")


def append_tensors_to_lists(list_of_lists, list_of_tensors):
    # Go through each tensor and corresponding list
    for i in range(len(list_of_tensors)):
        # Convert tensor to numpy and append it to the corresponding list
        list_of_lists[i] += [list_of_tensors[i].cpu().numpy()]
    # Return the lists
    return list_of_lists


def concatenate_lists(list_of_lists):
    list_of_np_arrs = []
    # Pick a list of numpy arrays ([np_arr1, np_arr2, ...]), concatenate numpy arrs to a single one (np_arr_big),
    # and append it back to the list ([np_arr_big1, np_arr_big2, ...])
    for list_ in list_of_lists:
        list_of_np_arrs.append(np.concatenate(list_))
    # Return numpy arrays
    return list_of_np_arrs
