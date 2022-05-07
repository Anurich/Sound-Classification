import os
import sys



ABS_PATH = os.path.abspath(".")
CSV_FILE_PATH = os.path.join(ABS_PATH, "data", "UrbanSound8K.csv")
CSV_TRAIN_FILE_PATH = os.path.join(ABS_PATH, "data", "train.csv")
CSV_TEST_FILE_PATH  = os.path.join(ABS_PATH,"data", "test.csv")
CSV_DEV_FILE_PATH   = os.path.join(ABS_PATH,"data", "dev.csv")
PICKLE_FILE_PATH_w2i    = os.path.join(ABS_PATH, "data", "word2index.pickle")
PICKLE_FILE_PATH_i2w    = os.path.join(ABS_PATH, "data", "index2word.pickle")
AUDIO_PATH  = os.path.join(ABS_PATH, "data", "audio")

SAMPLE_RATE  = 22050
HOP_LENGTH  = 1024
N_FFTT = 2048
N_MFCC = 40
BATCH_SIZE = 8
BATCH_SIZE_TEST = 8
LR  = 1e-5
ITERATION  = 10000
NUM_SAMPLES = 44050

# save features files

FEAT_PATH_TRAIN = os.path.join(ABS_PATH, "data", "features","train.pickle")
FEAT_PATH_DEV = os.path.join(ABS_PATH, "data",  "features","dev.pickle")
FEAT_PATH_TEST = os.path.join(ABS_PATH, "data", "features","test.pickle")

# model path 
MODEL_PATH  = os.path.join(ABS_PATH, "models")
sys.path.insert(0,MODEL_PATH)

MODEL_SAVED = "savemodel"
MODEL_SAVED = os.path.join(MODEL_SAVED,"model.pth" )