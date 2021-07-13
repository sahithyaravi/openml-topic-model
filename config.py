
# Dataset is already present in data folder. Set to True only if you want to download from OpenML again
DOWNLOAD_DATASET_AGAIN = False

# Pre-Processing dataset
PROCESS_DATASET = False
PROCESSING_PARTS_OF_SPEECH_INCLUDED = ['NOUN', 'VERB']

# Model Training
GRID_SEARCH = False  # Grid Search for best Model for this setting.
FILTER_WORDS = 0.6  # 0.8 means filter words repeated in more than 80% of documents
FINAL_MODEL = True  # Generate final model with best params from grid search

