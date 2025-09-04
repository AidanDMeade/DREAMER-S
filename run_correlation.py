""" define ratio/intensities """

from biospecml.visualisations.plotting_functions import plot_df
import matplotlib as mpl
import pandas as pd
import os
import numpy as np

ranges = {
    "Lipids (CO-O-C)":(1070,1170),
    "Amide I": (1600, 1700),
    "Amide II": (1540, 1560), # (1500, 1580), <- Amide II beta helix
    "Amide III": (1200, 1350), #(1200, 1350)
    "RNA (C=O)": (1720,1745),
    "Nucleic Acids (PO2−)":(1230,1244),
    "Lipid (C=O)":(1720, 1745),
    }

peaks = {
    "Phosphate ion (DNA)":965,
    "Amide I Peak": 1658,
    "Amide II Peak": 1544,
    "Amide III Peak": 1240,
    "Lipid/Protein (CH2) Peak (paraffin)": 1467,
    }

ratios = {
    "Ratio Amide I/II": (1652, 1544),
    "Ratio Amide I/DNA (PO4−)":(1652, 964),
    "Ratio Amide I/RNA (Uracil)":(1652, 996),
    "Ratio Amide I/RNA (C=O)":(1652, "RNA (C=O)"),
    "Ratio Amide I/Nucleic Acids (PO2−)":(1652, "Nucleic Acids (PO2−)"),
    "Ratio Amide I/Lipid (C=O)":(1652, "Lipid (C=O)"),
    "Ratio Amide I/Lipid (CO-O-C)":(1652, "Lipids (CO-O-C)"),
    }

PLOT_SAVEDIR = os.path.join('.', 'results', 'correlations')

metadata_cols = [] # declare metadata columns name
df_f = pd.DataFrame(spectral_data) # define spectral data
df_protein2 = pd.DataFrame(protein_data) # define the western blot data

# prep the data
X = df_f.drop(metadata_cols, axis=1)
X.columns = pd.to_numeric(X.columns)

def get_spectral_data_in_range(df, start_wn, end_wn):
    # Function to get wavenumbers and intensities within a specific range
    relevant_wavenumbers = [col for col in df.columns if start_wn <= col <= end_wn]
    return df[relevant_wavenumbers]


def calculate_area(intensities, wavenumbers):
    # Function to calculate area under the curve (using trapezoidal rule for simplicity)
    if len(intensities) <= 1: 
        return 0.0 # Handle cases with zero or one data point
    return np.trapz(intensities, wavenumbers)

# ------------------------------------------------------------------------------
# Calculate areas of each ranges for each sample

range_areas = {}
for band, rangei in ranges.items():
    
    lower, upper = rangei[0], rangei[1]
    range_areas[band] = {}

    for index, row in X.iterrows():
    
        # Get relevant data for the range
        range_data = get_spectral_data_in_range(pd.DataFrame([row]), lower, upper)
        range_data_wn_subset = range_data.columns.tolist()
        range_data_intensities_subset = range_data.iloc[0].values

        # Calculate areas
        range_area = calculate_area(range_data_intensities_subset, range_data_wn_subset)

        # Append data
        range_areas[band][index] = range_area

# update the table
band_cols = []
for band, data in range_areas.items():
    df_f[band] = data
    X[band] = data
    metadata_cols.append(band)
    band_cols.append(band)

print("Generated columns:", band_cols)

# ------------------------------------------------------------------------------
# Calculate ratio of each ranges for each sample

peak_ratios = {}
for band, peaks in ratios.items():
    print("Calculating ratio", band)
    peak1, peak2 = peaks[0], peaks[1]
    peak_ratios[band] = X[peak1]/X[peak2]

# update the table
for band, data in peak_ratios.items():
    df_f[band] = data
    X[band] = data
    metadata_cols.append(band)
    band_cols.append(band)

print("Generated columns:", band_cols)

# ------------------------------------------------------------------------------

""" calculate means of bands for each samples """

df_means = pd.DataFrame()

for sampleid, df_ in df_f.groupby("SampleID"):
    
    # get the majority kmeans newgroup labels 
    newgroup_km = df_["KMeansNewGroup"].unique()
    if len(newgroup_km)==1:
        newgroup_km = newgroup_km[0]
    elif len(newgroup_km)>1:
        newgroup_km = df_["KMeansNewGroup"].value_counts().idxmax()
    newgroup = df_["NewGroup"].unique()[0]
    
    # calculate the mean of ratios/inten 
    df_ = df_[band_cols]
    means = df_.mean().to_frame().T
    
    # append additional label
    means["NewGroup"] = newgroup_km
    df_means = pd.concat([df_means, means], axis=0)
    # break

df_means = df_means.sort_values(by="NewGroup")
df_means[['Patient', 'Treatment']] = df_means['NewGroup'].str.split('_', expand=True)
df_means.reset_index(inplace=True, drop=True)
print(df_means.shape)

""" rearrange protein df for correlation df """

proteindict = {}
for patient in df_protein2["Patient"].unique():
    df_protein2_ = df_protein2[df_protein2["Patient"]==patient]
    df_protein_rearr = pd.DataFrame()
    for newgroup, df_ in df_protein2_.groupby("NewGroup"):
        df_.drop(["Treatment"], axis=1, inplace=True)
        df_ = df_.rename(columns={"NewGroup": "Treatment"})
        df_rearr = pd.DataFrame()
        for protein, df__ in df_.groupby("Protein"):
            patient = df__["Patient"].unique()[0]
            df__ = df__.rename(columns={"ProteinValue": f"{protein} ({patient})"})
            df__.drop(["Protein", "Patient", "Treatment"], inplace=True, axis=1)
            df__.reset_index(inplace=True, drop=True)
            df_rearr = pd.concat([df_rearr, df__], axis=1)
        new_index = np.tile(df_["Treatment"].unique(), len(df_rearr))
        df_rearr.set_index(new_index, inplace=True)
        df_protein_rearr = pd.concat([df_protein_rearr, df_rearr], axis=0)
    print(patient, df_protein_rearr.shape)
    proteindict[patient] = df_protein_rearr

# =============================================================================


""" combine spectra with protein df according to patient """

patient = 'CRC0344' # PICK ONE! AND RUN THE REST
# patient = 'CRC0076'

# prep data
df1 = df_means[df_means["Patient"]==patient]
df2 = proteindict[patient].copy()
df2.reset_index(inplace=True)
df2 = df2.rename(columns={'index':'NewGroup'})
print("Length dfs inputs:", len(df1), len(df2))

# ------------------------------------------------------------------------------
# combine the two dfs (both dfs will have different length values)

df1_temp = df1.copy()
df2_temp = df2.copy()

# Create the helper key using groupby() and cumcount()
df1_temp['merge_key'] = df1_temp.groupby('NewGroup').cumcount()
df2_temp['merge_key'] = df2_temp.groupby('NewGroup').cumcount()

df_corr = df1_temp.merge(df2_temp, how='outer', on=['NewGroup', 'merge_key'])
df_corr = df_corr.drop('merge_key', axis=1)
print(df_corr.shape)
print(df_corr)

print("STATUS: Finished.")
