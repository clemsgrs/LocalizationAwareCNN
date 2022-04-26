# LocalizationAwareCNN
Personal research project

# How to run code?
To run training and testing, just download the `train_test.ipynb` notebook and open it with Google Colab. In the notebook, we first download the data, then we run training and testing.

# TODO:
- [ ] sparse tensor generation: try to benefit from GPU / multi thread on CPU to go FASTER!
- [ ] deal with fixed size inputs to CNN: get maximum slide dimension M (e.g. M=11000) & make all tensors of size np.ceil(M/224) (e.g. 50*50) (other possible solution: use GAP & enfore minimum tensor size to be consistent with CNN feature maps size (e.g. enforce 32x32 minimum?))
- [ ] deal with tissue filter: 1) check x,y coordinates are not inverted ; 2) check coordinates have not been translated by a given (tx, ty) vector (i.e. first tissue tile coords are (0,0))
- [ ] create data downloading script using Google Drive link
- [ ] maybe restrict myself to ROI which might all be the same size (+full of tisse +smaller size dataset)
- [ ] how about pre-training on ROIs, then use pre-trained weights to embed WSIs' tiles? This is because how the slide-level label was acquired (it's derived as the most severe cancerous lesion detected within the slide)

# Tests:

- [ ] check that tiles_df returns same dataframe when slide made full of tissue (e.g. extract a patch of tissue) with tissue_filter=True and tissue_filter=False