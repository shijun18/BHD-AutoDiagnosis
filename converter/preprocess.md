### How to pre-process the raw data from the hospital

The core idea of the pre-process is to convert `dcm` data to the specific format used for training. It consists of the following **six** steps:

- step 1 :  To remove the privacy information like the patient name, see `data_desens.modify_file_name`.
- Step 2 : Convert `dcm` series to `nii` format to further desensitize, see `data_desens.dcm_to_nii`.
- Step 3 : Get data attributes to guide the following process, such as resample and resize, see `data_desens.get_data_attr`.
- Step 4 : Given the negative effect of different pixel spacing of the data, all data are resampled to the fixed pixel spacing [1.,1.,1,], see `nii2npy.resample_nii`
- Step 5 : Convert the `nii` to `hdf5` (or the others) format after cropping used for training, see `nii2npy.nii2npy.`
- Step 6 : Generate data index and save as `csv` file used for training, see `generate_index.make_label_csv`
