######################################################################################################################
##########################   make topological features and put them into a hdf5 file #################################
######################################################################################################################
##########################   QM8    QM8     QM8     QM8     QM8     QM8     QM8     ##################################
######################################################################################################################

import sys
sys.path.append(r"C:\Users\ella_\Documents\GitHub\graphs_and_topology_for_chemistry")
sys.path.append(r"C:\Users\ella_\Documents\GitHub\icosahedron_projection")

import deepchem as dc

## fix cos their data was dodgy

dc.molnet.load_function.qm8_datasets.QM8_CSV_URL = "https://gist.githubusercontent.com/ellagale/9e632c9fcf11aaa803ae8a86f9c2a82d/raw/dd862e24858fe68141021c843069ecd0350b5d13/qm8.csv"

import tensorflow as tf
import os
import sys
import rdkit
import h5py
import helper_functions as h

from csv import reader
import csv
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

print("TensorFlow version: " + tf.__version__)

############################## cahnge this ###############################
#
save_dir=r'F:\Nextcloud\science\Datasets\topol_datasets'
data_dir=r'F:\Nextcloud\science\Datasets'
results_dir=r"F:\Nextcloud\science\results\topology_and_graphs\QM8"
#test_file='qm8.csv'
out_file_name='qm8_topological_features.hdf5'
smiles_file = 'qm8_SMILES.csv'
make_dataset=True # whether to recalc the dataset
make_hdf5 = True

print(f"DeepChem version: {dc.__version__}")
#
feature_name_list = ['pers_S_1', 'pers_S_2', 'pers_S_3',
                    'no_p_1', 'no_p_2', 'no_p_3',
                    'bottle_1', 'bottle_2', 'bottle_3',
                    'wasser_1', 'wasser_2', 'wasser_3',
                    'landsc_1', 'landsc_2', 'landsc_3',
                    'pers_img_1', 'pers_img_2', 'pers_img_3']
num_of_topol_features = len(feature_name_list)

#### Load data with no featurization

# This loads the data without doing any featurization
tasks, datasets, transformers = dc.molnet.load_qm8(
    shard_size=2000,
    featurizer=h.My_Dummy_Featurizer(None),
    splitter="index") # not shuffled

train_dataset, valid_dataset, test_dataset = datasets
num_of_molecules: int = len(train_dataset) + len(valid_dataset) + len(test_dataset)
print(f"Found {num_of_molecules} molecules")

# RUN THIS
#make_dataset = False
do_specified_range = True
selected_range = [x for x in range(1, 10)]
#num_of_molecules_to_do = len(selected_range)  ## change this to 0 to do all of them
num_of_molecules_to_do: int = 5#0 # to go into functions, 0 is an override to do all

current_ptr = 0
batch_size = 100
#remaining = num_of_molecules
## make smiles strings! and do unnormed


## we have to do this as the sdf file is missing 4 datapoints, so we need all the data
## to be taken directly from the deepchem dataset

# open the file in the write mode
f = open(os.path.join(save_dir, smiles_file), 'w', newline='')

# create the csv writer
writer = csv.writer(f)
print('Writing SMILES strings out')
dataset = train_dataset
for molecule_num in range(len(dataset)):
    row = [dataset.ids[molecule_num]]
    # print(row)
    # write a row to the csv file
    writer.writerow(row)

dataset = valid_dataset
for molecule_num in range(len(dataset)):
    row = [dataset.ids[molecule_num]]
    # print(row)
    # write a row to the csv file
    writer.writerow(row)

dataset = test_dataset
for molecule_num in range(len(dataset)):
    row = [dataset.ids[molecule_num]]
    # print(row)
    # write a row to the csv file
    writer.writerow(row)

    # close the file
f.close()

#######################################################################################################################
#                       Calculate the topological features and jot them down                                          #
#######################################################################################################################

#make_dataset = False
if make_dataset:
    with open(os.path.join(save_dir,'x_data_qm8.csv'), 'w') as f:
        with open(os.path.join(save_dir, 'y_data_qm8.csv'), 'w') as y_fh:
            # train
            remaining = len(train_dataset)
            h.temp_write_topol_data(
                f,
                remaining=remaining,
                current_ptr=0,
                my_dataset=train_dataset,
                num_of_topol_features=num_of_topol_features,
                do_specified_range=do_specified_range,
                batch_size=batch_size,
                data_dir=data_dir
            )
            untransformed_train_y = transformers[0].untransform(train_dataset.y)
            h.copy_targets_into_csv(
                y_fh=y_fh,
                my_dataset=untransformed_train_y
            )
            # validate
            remaining = len(valid_dataset)
            h.temp_write_topol_data(
                f,
                remaining=remaining,
                current_ptr=0,
                my_dataset=valid_dataset,
                num_of_topol_features=num_of_topol_features,
                do_specified_range=do_specified_range,
                batch_size=batch_size,
                data_dir=data_dir
            )
            untransformed_valid_y = transformers[0].untransform(valid_dataset.y)
            h.copy_targets_into_csv(
                y_fh=y_fh,
                my_dataset=untransformed_valid_y
            )
            # test
            remaining = len(test_dataset)
            h.temp_write_topol_data(
                f,
                remaining=remaining,
                current_ptr=0,
                my_dataset=test_dataset,
                num_of_topol_features=num_of_topol_features,
                do_specified_range=do_specified_range,
                batch_size=batch_size,
                data_dir=data_dir
            )
            untransformed_test_y = transformers[0].untransform(test_dataset.y)
            h.copy_targets_into_csv(
                y_fh=y_fh,
                my_dataset=untransformed_test_y
            )
    f.close()

#sys.exit(0)
##################################################################################################################
#                               load data                                                                        #
##################################################################################################################
topl_features_df = pd.read_csv(os.path.join(save_dir,'x_data_qm8.csv'))
topl_targets_df = pd.read_csv(os.path.join(save_dir,'y_data_qm8.csv'))

topl_QM8_all_mat=topl_features_df

qm8_df=pd.read_csv(os.path.join(save_dir,'qm8_SMILES.csv')) # has smiles strings

with open(os.path.join(save_dir,'x_data_qm8.csv'), 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    topl_QM8_all_list = list(csv_reader)
    print(topl_QM8_all_list)

topl_QM8_all_list = np.array(topl_QM8_all_list)

with open(os.path.join(save_dir,'y_data_qm8.csv'), 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    targets_topl_QM8_all_list = list(csv_reader)
    print('Example y data')
    print(targets_topl_QM8_all_list[0])

with open(os.path.join(save_dir,'qm8_SMILES.csv'), 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    SMILES_list = list(csv_reader)

################# now do the PCA ###################################################################################
#pca = PCA(n_components=num_of_topol_features)
pca = PCA(n_components=num_of_topol_features)
principalComponents_large = pca.fit_transform(topl_QM8_all_list)
####################################################################################################################
#SMILES_list = qm7_df['smiles']

mol_idx = 0

if make_hdf5:
    outfile = h5py.File(os.path.join(save_dir,out_file_name),"w")
    string_type = h5py.string_dtype(encoding='utf-8')

    num_of_molecules_override=0

    if True:

        if num_of_molecules_override == 0:
            # do all proteins woo
            num_of_molecules_to_do= num_of_molecules

        else:
            num_of_molecules_to_do = num_of_molecules_override
        print(f'Processing {num_of_molecules} molecules')
        ##################### set up the output datasets ################################

        ## this sets up the output datasets
        molID_ds = h.create_or_recreate_dataset(outfile, "molID", (num_of_molecules_to_do,), dtype=np.int8)
        SMILES_ds = h.create_or_recreate_dataset(outfile, "SMILES", (num_of_molecules_to_do,), dtype=string_type)
            #                                       ##### topological data ###                              #
        ###### proteins #####
        #      Persistence entropy
        P_pers_S_1_ds = h.create_or_recreate_dataset(outfile, "pers_S_1", (num_of_molecules_to_do,), dtype=np.float32)
        P_pers_S_2_ds = h.create_or_recreate_dataset(outfile, "pers_S_2", (num_of_molecules_to_do,), dtype=np.float32)
        P_pers_S_3_ds = h.create_or_recreate_dataset(outfile, "pers_S_3", (num_of_molecules_to_do,), dtype=np.float32)
        #      No. of points
        P_no_p_1_ds = h.create_or_recreate_dataset(outfile, "no_p_1", (num_of_molecules_to_do,), dtype=np.float32)
        P_no_p_2_ds = h.create_or_recreate_dataset(outfile, "no_p_2", (num_of_molecules_to_do,), dtype=np.float32)
        P_no_p_3_ds = h.create_or_recreate_dataset(outfile, "no_p_3", (num_of_molecules_to_do,), dtype=np.float32)
        #      Bottleneck
        P_bottle_1_ds = h.create_or_recreate_dataset(outfile, "bottle_1", (num_of_molecules_to_do,), dtype=np.float32)
        P_bottle_2_ds = h.create_or_recreate_dataset(outfile, "bottle_2", (num_of_molecules_to_do,), dtype=np.float32)
        P_bottle_3_ds = h.create_or_recreate_dataset(outfile, "bottle_3", (num_of_molecules_to_do,), dtype=np.float32)
        #      Wasserstein
        P_wasser_1_ds = h.create_or_recreate_dataset(outfile, "wasser_1", (num_of_molecules_to_do,), dtype=np.float32)
        P_wasser_2_ds = h.create_or_recreate_dataset(outfile, "wasser_2", (num_of_molecules_to_do,), dtype=np.float32)
        P_wasser_3_ds = h.create_or_recreate_dataset(outfile, "wasser_3", (num_of_molecules_to_do,), dtype=np.float32)
        #      landscape
        P_landsc_1_ds = h.create_or_recreate_dataset(outfile, "landsc_1", (num_of_molecules_to_do,), dtype=np.float32)
        P_landsc_2_ds = h.create_or_recreate_dataset(outfile, "landsc_2", (num_of_molecules_to_do,), dtype=np.float32)
        P_landsc_3_ds = h.create_or_recreate_dataset(outfile, "landsc_3", (num_of_molecules_to_do,), dtype=np.float32)
        #      persistence image
        P_pers_img_1_ds = h.create_or_recreate_dataset(outfile, "pers_img_1", (num_of_molecules_to_do,), dtype=np.float32)
        P_pers_img_2_ds = h.create_or_recreate_dataset(outfile, "pers_img_2", (num_of_molecules_to_do,), dtype=np.float32)
        P_pers_img_3_ds = h.create_or_recreate_dataset(outfile, "pers_img_3", (num_of_molecules_to_do,), dtype=np.float32)

            #                                       PCs                                                 #
        PCA_1_ds = h.create_or_recreate_dataset(outfile, "PCA_1", (num_of_molecules_to_do,), dtype=np.float32)
        PCA_2_ds = h.create_or_recreate_dataset(outfile, "PCA_2", (num_of_molecules_to_do,), dtype=np.float32)
        PCA_3_ds = h.create_or_recreate_dataset(outfile, "PCA_3", (num_of_molecules_to_do,), dtype=np.float32)
        PCA_4_ds = h.create_or_recreate_dataset(outfile, "PCA_4", (num_of_molecules_to_do,), dtype=np.float32)
        PCA_5_ds = h.create_or_recreate_dataset(outfile, "PCA_5", (num_of_molecules_to_do,), dtype=np.float32)
        PCA_6_ds = h.create_or_recreate_dataset(outfile, "PCA_6", (num_of_molecules_to_do,), dtype=np.float32)
        PCA_7_ds = h.create_or_recreate_dataset(outfile, "PCA_7", (num_of_molecules_to_do,), dtype=np.float32)
        PCA_8_ds = h.create_or_recreate_dataset(outfile, "PCA_8", (num_of_molecules_to_do,), dtype=np.float32)
        PCA_9_ds = h.create_or_recreate_dataset(outfile, "PCA_9", (num_of_molecules_to_do,), dtype=np.float32)
        PCA_10_ds = h.create_or_recreate_dataset(outfile, "PCA_10", (num_of_molecules_to_do,), dtype=np.float32)
        PCA_11_ds = h.create_or_recreate_dataset(outfile, "PCA_11", (num_of_molecules_to_do,), dtype=np.float32)
        PCA_12_ds = h.create_or_recreate_dataset(outfile, "PCA_12", (num_of_molecules_to_do,), dtype=np.float32)
        PCA_13_ds = h.create_or_recreate_dataset(outfile, "PCA_13", (num_of_molecules_to_do,), dtype=np.float32)
        PCA_14_ds = h.create_or_recreate_dataset(outfile, "PCA_14", (num_of_molecules_to_do,), dtype=np.float32)
        PCA_15_ds = h.create_or_recreate_dataset(outfile, "PCA_15", (num_of_molecules_to_do,), dtype=np.float32)
        PCA_16_ds = h.create_or_recreate_dataset(outfile, "PCA_16", (num_of_molecules_to_do,), dtype=np.float32)
        PCA_17_ds = h.create_or_recreate_dataset(outfile, "PCA_17", (num_of_molecules_to_do,), dtype=np.float32)
        PCA_18_ds = h.create_or_recreate_dataset(outfile, "PCA_18", (num_of_molecules_to_do,), dtype=np.float32)
            # # #                           targets                                         # # #
        target_E1_CC2_ds = h.create_or_recreate_dataset(outfile, "E1-CC2", (num_of_molecules_to_do,), dtype=np.float32)
        target_E2_CC2_ds = h.create_or_recreate_dataset(outfile, "E2-CC2", (num_of_molecules_to_do,), dtype=np.float32)
        target_f1_CC2_ds = h.create_or_recreate_dataset(outfile, "f1-CC2", (num_of_molecules_to_do,), dtype=np.float32)
        target_f2_CC2_ds = h.create_or_recreate_dataset(outfile, "f2-CC2", (num_of_molecules_to_do,), dtype=np.float32)
        target_E1_PBE0_ds = h.create_or_recreate_dataset(outfile, "E1-PBE0", (num_of_molecules_to_do,), dtype=np.float32)
        target_E2_PBE0_ds = h.create_or_recreate_dataset(outfile, "E2-PBE0", (num_of_molecules_to_do,), dtype=np.float32)
        target_f1_PBE0_ds = h.create_or_recreate_dataset(outfile, "f1-PBE0", (num_of_molecules_to_do,), dtype=np.float32)
        target_f2_PBE0_ds = h.create_or_recreate_dataset(outfile, "f2-PBE0", (num_of_molecules_to_do,), dtype=np.float32)
        target_E1_PBE0_2_ds = h.create_or_recreate_dataset(outfile, "E1-PBE0_2", (num_of_molecules_to_do,), dtype=np.float32)
        target_E2_PBE0_2_ds = h.create_or_recreate_dataset(outfile, "E2-PBE0_2", (num_of_molecules_to_do,), dtype=np.float32)
        target_f1_PBE0_2_ds = h.create_or_recreate_dataset(outfile, "f1-PBE0_2", (num_of_molecules_to_do,), dtype=np.float32)
        target_f2_PBE0_2_ds = h.create_or_recreate_dataset(outfile, "f2-PBE0_2", (num_of_molecules_to_do,), dtype=np.float32)
        target_E1_CAM_ds = h.create_or_recreate_dataset(outfile, "E1-CAM", (num_of_molecules_to_do,), dtype=np.float32)
        target_E2_CAM_ds = h.create_or_recreate_dataset(outfile, "E2-CAM", (num_of_molecules_to_do,), dtype=np.float32)
        target_f1_CAM_ds = h.create_or_recreate_dataset(outfile, "f1-CAM", (num_of_molecules_to_do,), dtype=np.float32)
        target_f2_CAM_ds = h.create_or_recreate_dataset(outfile, "f2-CAM", (num_of_molecules_to_do,), dtype=np.float32)


        for mol_idx in range(num_of_molecules_to_do):
            if mol_idx % 50 == 0:
                print('Got to Molecule no. ', mol_idx, SMILES_list[mol_idx])
            molID_ds[mol_idx] = mol_idx
            # get the current PDB code
            #current_code=PDB_List[mol_idx]
            # get the rows in the dataframes
            #current_index_row=df_index.loc[df_index['PDB_code']==current_code]
            #print(current_code)
            #print(current_index_row)
            #current_data_row=df_data.loc[df_data['PDB_code']==current_code]
            #print(current_data_row)

            smiles_string = SMILES_list[mol_idx]
            SMILES_ds[mol_idx] = np.array(smiles_string, dtype=string_type)

                #                   toplogical features             #
            (P_pers_S_1_ds[mol_idx], P_pers_S_2_ds[mol_idx], P_pers_S_3_ds[mol_idx],
            P_no_p_1_ds[mol_idx], P_no_p_2_ds[mol_idx],P_no_p_3_ds[mol_idx],
            P_bottle_1_ds[mol_idx],P_bottle_2_ds[mol_idx],P_bottle_3_ds[mol_idx],
            P_wasser_1_ds[mol_idx],P_wasser_2_ds[mol_idx], P_wasser_3_ds[mol_idx],
            P_landsc_1_ds[mol_idx], P_landsc_2_ds[mol_idx],P_landsc_3_ds[mol_idx],
            P_pers_img_1_ds[mol_idx],P_pers_img_2_ds[mol_idx],P_pers_img_3_ds[mol_idx]
             ) = np.array(topl_QM8_all_list[mol_idx], dtype='f8')
                #                           PCs                         #
            (PCA_1_ds[mol_idx], PCA_2_ds[mol_idx], PCA_3_ds[mol_idx], PCA_4_ds[mol_idx], PCA_5_ds[mol_idx],
            PCA_6_ds[mol_idx], PCA_7_ds[mol_idx], PCA_8_ds[mol_idx], PCA_9_ds[mol_idx], PCA_10_ds[mol_idx],
            PCA_11_ds[mol_idx], PCA_12_ds[mol_idx], PCA_13_ds[mol_idx], PCA_14_ds[mol_idx], PCA_15_ds[mol_idx],
            PCA_16_ds[mol_idx], PCA_17_ds[mol_idx], PCA_18_ds[mol_idx])=principalComponents_large[mol_idx]

            # targets
            # # #                           targets                                         # # #
            target_E1_CC2_ds[mol_idx] = np.array(targets_topl_QM8_all_list[mol_idx][0], dtype='f8')
            target_E2_CC2_ds[mol_idx] = np.array(targets_topl_QM8_all_list[mol_idx][1], dtype='f8')
            target_f1_CC2_ds[mol_idx] = np.array(targets_topl_QM8_all_list[mol_idx][2], dtype='f8')
            target_f2_CC2_ds[mol_idx] = np.array(targets_topl_QM8_all_list[mol_idx][3], dtype='f8')
            target_E1_PBE0_ds[mol_idx] = np.array(targets_topl_QM8_all_list[mol_idx][4], dtype='f8')
            target_E2_PBE0_ds[mol_idx] = np.array(targets_topl_QM8_all_list[mol_idx][5], dtype='f8')
            target_f1_PBE0_ds[mol_idx] = np.array(targets_topl_QM8_all_list[mol_idx][6], dtype='f8')
            target_f2_PBE0_ds[mol_idx] = np.array(targets_topl_QM8_all_list[mol_idx][7], dtype='f8')
            target_E1_PBE0_2_ds[mol_idx] = np.array(targets_topl_QM8_all_list[mol_idx][8], dtype='f8')
            target_E2_PBE0_2_ds[mol_idx] = np.array(targets_topl_QM8_all_list[mol_idx][9], dtype='f8')
            target_f1_PBE0_2_ds[mol_idx] = np.array(targets_topl_QM8_all_list[mol_idx][10], dtype='f8')
            target_f2_PBE0_2_ds[mol_idx] = np.array(targets_topl_QM8_all_list[mol_idx][11], dtype='f8')
            target_E1_CAM_ds[mol_idx] = np.array(targets_topl_QM8_all_list[mol_idx][12], dtype='f8')
            target_E2_CAM_ds[mol_idx] = np.array(targets_topl_QM8_all_list[mol_idx][13], dtype='f8')
            target_f1_CAM_ds[mol_idx] = np.array(targets_topl_QM8_all_list[mol_idx][14], dtype='f8')
            target_f2_CAM_ds[mol_idx] = np.array(targets_topl_QM8_all_list[mol_idx][15], dtype='f8')


    outfile.close()

sys.exit(0)






