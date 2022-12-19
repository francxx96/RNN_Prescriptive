from pathlib import Path

import shared_variables
from matplotlib import pyplot as plt


def create_checkpoints_path(log_name, models_folder, fold, model_type):
    folder_path = shared_variables.outputs_folder / models_folder / str(fold) / 'models' / model_type / log_name
    if not Path.exists(folder_path):
        Path.mkdir(folder_path, parents=True)
    checkpoint_name = folder_path / 'model_{epoch:03d}-{val_loss:.3f}.h5'
    return str(checkpoint_name)


def plot_loss(history, dir_name):
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(str(Path(dir_name) / "loss.png"))
