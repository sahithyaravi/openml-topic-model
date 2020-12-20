# openml-topic-model
 
We have about 40,000 datasets on OpenML.  We would like to group these datasets into topics, based on the description of the datasets.

In this repo:
- The data folder contains the latest version of the downloaded descriptions.
- The src folder has the source code for obtaining the dataset descriptions (getdata.py),
preprocessing and creating a pre-processed dataframe(preprocess.py) and algorithms for performing 
topic modeling (model.py). utils.py and preprocess.py have helper functions which are used by the other files.

- The config.py files allows you to configure whether the dataset needs to be downloaded again (DOWNLOAD_DATASET_AGAIN),
whether it needs to be preprocessed again and also allows you to configure the preprocessing methods.
- Once the parameters are configured in config.py, the model can be run using run_model.py and the results should be available in the results folder.
- We currently support LDA with different parameters and seeded LDA. Support for contextualized topic models will be added soon.