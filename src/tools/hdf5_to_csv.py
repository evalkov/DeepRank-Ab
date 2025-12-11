import sys
import h5py
import pandas as pd
import numpy as np

# This helper function hdf5_to_csv converts the structured HDF5 log produced by your GNN into a flat 
# CSV table. In a nutshell:
def hdf5_to_csv(hdf5_path):
        
        # Open the HDF5 file in read/write mode.
        hdf5 = h5py.File(hdf5_path,'r+')
        # gets the name
        name = hdf5_path.split('.')[0]
        
        #  “we still need to write the CSV header line.” It’ll flip to False after the first write.
        first = True
        # loop over every epoch and subgroup under it (train eval test)
        for epoch in hdf5.keys():
                # Loop over splits
                for dataset in hdf5['{}'.format(epoch)].keys():
                        mol = hdf5['{}/{}/mol'.format(epoch, dataset)]
                        # Builds two Python lists of the same length as mol, filled with the epoch name 
                        # and dataset name. Later each row gets one entry.
                        epoch_lst = [epoch] * len(mol)
                        dataset_lst = [dataset] * len(mol)

                        # 
                        outputs = hdf5['{}/{}/outputs'.format(epoch, dataset)]
                        targets = hdf5['{}/{}/targets'.format(epoch, dataset)]
                        # If the “targets” array is empty, replace it with a dummy string of length N so 
                        # you still get one entry per molecule.
                        if len(targets) == 0:
                                targets = 'n'*len(mol)
                        # A second flag to detect whether we have multi-column “raw_outputs” 
                        # (per-class probabilities) below.
                        bin=False

                        # This section is specific to the classes                                                                                                                    
                        # it adds the raw output, i.e. probabilities to belong to the class 0, the class 1, etc., to the prediction hdf5                                             
                        # This way, binary information can be transformed back to continuous data and used for ranking                                                               
                        
                        # Checks if there’s a "raw_outputs" dataset in this epoch/split. If so, we’ll 
                        # extract per-class probabilities too.
                        if 'raw_outputs' in hdf5['{}/{}'.format(epoch, dataset)].keys():
                                # Loads the raw_outputs array into arr, and if it’s 2D or higher (shape (N, C)), sets 
                                # bin = True so we know to extract each class column separately.
                                if len(hdf5['{}/{}/raw_outputs'.format(epoch, dataset)][()].shape) > 1:
                                        bin=True
                                        # Write CSV header (once)
                                        if first :
                                                header = ['epoch', 'set', 'model', 'targets', 'prediction']
                                                output_file = open('{}.csv'.format(name), 'w')
                                                output_file.write(','+','.join(header)+'\n')
                                                output_file.close()
                                                first = False
                                        # Assemble a DataFrame with raw outputs
                                        data_to_save = [epoch_lst, dataset_lst, mol, targets, outputs]
                                        
                                        for target_class in range(0,len(hdf5['{}/{}/raw_outputs'.format(epoch, dataset)][()])):
                                                # probability of getting 0                                                                                                                   
                                                outputs_per_class = hdf5['{}/{}/raw_outputs'.format(epoch, dataset)][()][:,target_class]
                                                data_to_save.append(outputs_per_class)
                                                header.append(f'raw_prediction_{target_class}')
                                        dataset_df = pd.DataFrame(list(zip(*data_to_save)), columns = header)
                        # Fallback for non-multi-class
                        #If we never set bin=True (i.e. no multi-class raw_outputs), we still need a header (once) 
                        # and build a simpler DataFrame with just those five columns.
                        if bin==False:
                                if first :
                                        header = ['epoch', 'set', 'model', 'targets', 'prediction']
                                        output_file = open('{}.csv'.format(name), 'w')
                                        output_file.write(','+','.join(header)+'\n')
                                        output_file.close()
                                        first = False
                                dataset_df = pd.DataFrame(list(zip(epoch_lst, dataset_lst, mol, targets, outputs)), columns = header)
                        
                        # Append to CSV
                        dataset_df.to_csv('{}.csv'.format(name), mode='a', header=True)
        

# If you run the script directly, checks you provided exactly one argument.

#On success: calls hdf5_to_csv on that filename.

#On error: prints a user-friendly message.
if __name__ == "__main__":	
        
	if len(sys.argv) != 2 :
        	print ("""\n
This scripts converts the hdf5 output files of GraphProt into csv files
                
Usage: 
python hdf5_to_csv.py file.hdf5
""")
                
	else: 
                try: 
                        hdf5_path = sys.argv[1]
                        hdf5_to_csv(hdf5_path)
                        
                except:
                        print('Please make sure that your input file if a HDF5 file')
