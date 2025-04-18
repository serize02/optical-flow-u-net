import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data/processed/u-net')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'artifacts')

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, 'models')
PREDICTION_SAVE_DIR = os.path.join(OUTPUT_DIR, 'predictions')

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 15
INPUT_CHANNELS = 1  
OUTPUT_CHANNELS = 1

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(PREDICTION_SAVE_DIR, exist_ok=True)