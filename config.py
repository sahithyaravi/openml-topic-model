
# Dataset is already present in data folder. Set to True only if you want to download from OpenML again
DOWNLOAD_DATASET_AGAIN = False

# Pre-Processing dataset
PROCESS_DATASET = True
PROCESSING_PARTS_OF_SPEECH_INCLUDED = ['NOUN']

# Model Training
GRID_SEARCH = True  # Grid Search for best Model for this setting.
FILTER_WORDS = 0.8  # 0.8 means filter words repeated in more than 80% of documents
FINAL_MODEL = True  # Generate final model with best params from grid search

