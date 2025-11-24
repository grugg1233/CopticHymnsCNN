import os

class Config():
    #path information
    ROOT = "output"
    CHECKPOINTS = f"{ROOT}/checkpoints"
    LOGS = f"{ROOT}/logs"
    BEST_MODEL = f"{CHECKPOINTS}/best_hymn_cnn.pt"

    DATA_ROOT = "/kaggle/input/hymnologicaldata/data/hymns"

    #audio features
    SAMPLE_RATE=16_000
    CLIP_SEC = 5.0
    HOP_SEC = 2.5
    N_MELS = 64

    #model 
    N_CLASSES = 4
    DROPOUT_P = 0.5

    #training
    LR = 5e-4
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    NUM_WORKERS = 4
    SEED = 42 #for reproducibility

    LABEL_MAP = {
        "golgotha": 0,
        "jenainan": 1,
        "taishori": 2,
        "tishori": 3,

    }


    @staticmethod
    def init_dirs():
        os.makedirs(Config.ROOT, exist_ok=True)
        os.makedirs(Config.CHECKPOINTS, exist_ok=True)
        os.makedirs(Config.LOGS, exist_ok=True)