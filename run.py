"""
Main python file for model optimisation training.
Change any training parameter in the 'training' section.
Install biospecml module first according to instruction at:
https://github.com/rafsanlab/BioSpecML

Author M. R. Rafsanjani @rafsanlab

--------------------------------
Directory structure:
main/
    data/
        A.MAT
        B.MAT
        C.MAT
        ...
    metadata/
        metadata-pdx_train.tsv
        metadata-pdx_test.tsv
        mean-std_train.pt
    src/
        X.py
        Y.py
        ...
    run.py
--------------------------------
"""

# ==============================================================================
#                                    IMPORTS
# ==============================================================================

import biospecml
import os, sys
import subprocess

work_dir = os.path.join(os.getcwd(), 'src')
sys.path.append(work_dir)

repo_url = "https://github.com/rafsanlab/messylib.git"
clone_dir = os.path.join(work_dir, "messylib")
if 'messylib' not in list(os.listdir(work_dir)):
    subprocess.run(["git", "clone", repo_url, clone_dir], check=True)
    requirements_path = os.path.join(clone_dir, 'messylib/commons/requirements.txt')
    subprocess.run(["pip", "install", "-r", requirements_path],check=True)

from biospecml.training_loops.train_loop_v2 import train_val_loop
from messylib.messylib.commons.files import get_filepaths_bytype
from ChemImgDatasetV0 import read_metadata, labels_dict, ChemImgDataset
from biospecml.models.LinearNet2 import LinearNet
import torch.optim as optim
import torch.nn as nn
import torch
import pandas as pd
from itertools import product
import gc

# ==============================================================================
#                                  PARAMETERS
# ==============================================================================

# set up paths
fold_metadata_path = {
    "1": {
        "savedir": os.path.join(os.getcwd(), "results/model-opt-results/"),
        "train": os.path.join(os.getcwd(), "metadata/metadata-pdx_train.tsv"),
        "val.": os.path.join(os.getcwd(), "metadata/metadata-pdx_test.tsv"),
        "mean-std": os.path.join(os.getcwd(), "metadata/mean-std_train.pt"),
        }
    }
    
chemimg_path = os.path.joiin(os.getcwd(), "data/")
label_col = "Patient"
id_col = "SampleID"

batch_size = 2
check_data = True

# build a hyperparameter grid
param_grid = {
    "num_blocks": [1,2,3],
    "residual_mode": [True, False],
    "hidden_expansion": ["double", "half"],
    "hidden_size": [64, 128, 256],
    }

# ==============================================================================
#                                  RUNNING
# ==============================================================================

# compile all the paths to the chem img into a files dict
chemimg_paths = get_filepaths_bytype(chemimg_path, ".mat")
chemimg_dict = {}
for path in chemimg_paths:
    fname = os.path.basename(path)
    posid = fname.split(".")[0]
    chemimg_dict[posid] = path
print(f"Total chem img files: {len(chemimg_dict)}")

# iterate each fold and train/val
for fold in fold_metadata_path.keys():

    print(f"Running for FOLD {fold}..")

    paths = fold_metadata_path[fold]

    # get the meta df and filtered files dict
    df_meta_train, chemimg_dict_train = read_metadata(
        metadata_path=paths["train"],
        filesdict=chemimg_dict,
        label_col=label_col,
        id_col=id_col,
    )
    df_meta_val, chemimg_dict_val = read_metadata(
        metadata_path=paths["val."],
        filesdict=chemimg_dict,
        label_col=label_col,
        id_col=id_col,
    )

    # get mean and std
    mean_std = torch.load(paths["mean-std"])
    mean, std = mean_std["mean"], mean_std["std"]

    # declare dataset
    train_dataset = ChemImgDataset(
        filesdict=chemimg_dict_train,
        metadata_df=df_meta_train,
        id_col=id_col,
        label_col=label_col,
        labels_dict=labels_dict,
        mean=mean,
        std=std,
        return_datadict=False,
    )
    val_dataset = ChemImgDataset(
        filesdict=chemimg_dict_val,
        metadata_df=df_meta_val,
        id_col=id_col,
        label_col=label_col,
        labels_dict=labels_dict,
        mean=mean,
        std=std,
        return_datadict=False,
    )

    # declare data loader
    train_loader = train_dataset.get_data_loader(
        batch_size=batch_size,
        shuffle=False,
        imbalance_data=False,
        return_imbalance_weight=False,
    )
    val_loader = val_dataset.get_data_loader(
        batch_size=batch_size,
        shuffle=False,
        imbalance_data=False,
    )
    
    # check data in data loaders
    if check_data:    
        for data, label in train_loader:
            print(data.shape, label.shape)
            break
        for data, label in val_loader:
            print(data.shape, label.shape)
            break
    
    # --------------------------------------------------------------------------
    # Hyperparameter tuning
    
    keys, values = zip(*param_grid.items())
    i = 1
    for p_values in product(*values):
        current_params = dict(zip(keys, p_values))
        print(f"Running param {i}:", current_params)
        i += 1
        
        paramname = ("_").join([str(v) for v in current_params.values()])
        paramdir = os.path.join(fold_metadata_path["1"]["savedir"], paramname)
        os.makedirs(paramdir)

        # declare model
        model = LinearNet(
            input_size=213,
            hidden_expansion=current_params["hidden_expansion"],
            hidden_size=current_params["hidden_size"],
            num_classes=2,
            dropout_rate=None,
            add_num_blocks=current_params["num_blocks"],
            residual_mode=current_params["residual_mode"],
            weakly_sv=True,
            mil_aggregation_mode="attention",
            use_softmax=False,
        )

        optimizer = optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=0
            )

        model, train_metrics = train_val_loop(
            model,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            num_epochs=20,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            train_loader=train_loader,
            test_loader=val_loader,
            trained_num_epochs=None,
            running_type='prediction',
            verbose=True,
            f1_average='macro',
            f1_average_test='macro',
            f1_zero_division=1, # 'warn', 1, or 0. recommended=0
            metrics_list=['f1', 'accuracy'],
            savedir=paramdir,
            epoch_save_checkpoints=[],
            save_model=True,
            use_lr_scheduler=False,
            lr_scheduler_step_size=None,
            lr_scheduler_gamma=None,
            )

        # clear memory
        del model
        del optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # break

print("STATUS: Finished")
