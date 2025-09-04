""" Run UMAP on the best attented spectra. """

from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, homogeneity_score, completeness_score
from sklearn.metrics import silhouette_score
import json
import umap
from itertools import product
import os
import pandas as pd

# ==============================================================================
#                                    PARAMETER
# ==============================================================================

RUNNING_MODE = "best" # "opt" or "best" <- IMPORTANT, use opt for optimisation

metrics = ['euclidean'] # ['euclidean', 'cosine', 'manhattan', 'correlation',] # ['euclidean'] 
n_neighbors = [80] # [5, 10, 20, 40, 60, 80] # [60]
min_dists = [0.25] # [0.1, 0.25, 0.5] # [0.1] 
n_components = 2
random_state = 107


# SAVEDIR = None
SAVEDIR = os.path.join(".", f"results/{RUNNING_MODE}")

label_col = "NewGroup" # define the group label
labeldict = { # default is None
    "CRC0076_ABT":0,
    "CRC0076_VEHICLE":1,
    "CRC0076_COMBO":2,
    "CRC0076_FOLFOX":3,
    "CRC0344_ABT":4,
    "CRC0344_VEHICLE":5,
    "CRC0344_COMBO":6,
    "CRC0344_FOLFOX":7,
    }

df_top = pd.DataFrame(data) # load data as the top spectra
metadata_cols = [] # declare metadata columns name


# ==============================================================================
#                                    RUNNING
# ==============================================================================

print(df_top[label_col].value_counts())
print(df_top["Patient"].value_counts())

if SAVEDIR:
    os.makedirs(SAVEDIR, exist_ok=True)

# convert labels
if labeldict is not None:
    labels = df_top[label_col].map(labeldict).to_list()
else:
    labels = df_top[label_col].to_list()

show_plot = False if RUNNING_MODE=="opt" else True
plot_filetypes = ["png"] if RUNNING_MODE=="opt" else ["png","svg"]

# ------------------------------------------------------------------------------
# run umap

# # scale data
X = df_top.drop(metadata_cols, axis=1).values
scaler = StandardScaler() # or use QuantileTransformer(output_distribution='normal')
X = scaler.fit_transform(X)

# run UMAP
all_results = {}
for metric, n_neighbor, min_dist in product(metrics, n_neighbors, min_dists):
    paramid = f"{metric}-{n_neighbor}-{min_dist}"
    print('Running paramid:', paramid)

    manifold = umap.UMAP(
        n_neighbors=n_neighbor,
        min_dist=min_dist, # default is 0.1
        n_components=n_components,
        metric=metric,
        random_state=random_state
        ).fit(X, labels)
    embedding = manifold.transform(X)
    # embedding = manifold.fit_transform(X)
    print(embedding.shape)

    # --------------------------------------------------------------------------
    # run k-means clustering
    
    kmeans = KMeans(
        n_clusters=len(df_top[label_col].unique()),
        random_state=random_state
        )
    predicted_labels = kmeans.fit_predict(embedding)
    df_top['KMeans'] = predicted_labels

    # check the clustering results
    true_labels = labels

    clustering_results = {
        'nmi' : float(normalized_mutual_info_score(true_labels, predicted_labels)),
        'hi' : float(homogeneity_score(true_labels, predicted_labels)),
        'ci' : float(completeness_score(true_labels, predicted_labels)),
        'sc' : float(silhouette_score(embedding, predicted_labels)),
        }
    all_results[paramid] = clustering_results

    output = "Clustering performance | "
    for test, results in clustering_results.items():
        output += f"{test} : {results:4f} | "
    print(output.strip(" | "))

# ------------------------------------------------------------------------------
# save results
if SAVEDIR:
    
    # save clustering results
    json_file = os.path.join(SAVEDIR, 'clustering-results.json')
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=4)

    # save the updated top att spectra tsv
    if RUNNING_MODE=="best":
        fname_ = os.path.join("results", "top-att-spectra_umap.tsv")
        df_top.to_csv(fname_, sep='\t')

print("STATUS: Finished.")
