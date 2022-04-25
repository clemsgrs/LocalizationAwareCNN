# LocalizationAwareCNN
Personal research project

# How to run code?
To run training and testing, just download the `train_test.ipynb` notebook and open it with Google Colab. In the notebook, we first download the data, then we run training and testing.

# TODO:
- [ ] deal with fixed size inputs to CNN: get maximum slide dimension M (e.g. M=11000) & make all tensors of size np.ceil(M/224) (e.g. 50*50)
- [ ] deal with tissue filter: 1) check x,y coordinates are not inverted ; 2) check coordinates have not been translated by a given (tx, ty) vector (i.e. first tissue tile coords are (0,0))
- [ ] create data downloading script using Google Drive link

# Tests:

- [ ] check that tiles_df returns same dataframe when slide made full of tissue (e.g. extract a patch of tissue) with tissue_filter=True and tissue_filter=False