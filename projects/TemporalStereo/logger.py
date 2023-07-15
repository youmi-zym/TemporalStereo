import os
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
import time

from torch.utils.collect_env import get_pretty_env_info
import PIL

def get_pil_version():
    return "\n        Pillow ({})".format(PIL.__version__)

def collect_env_info():
    env_str = get_pretty_env_info()
    env_str += get_pil_version()
    return env_str

def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)

class FileWriter:
    def __init__(self, save_path):
        self._save_path = save_path
        self.start_time = time.time()
        self.num_total_steps = 0

        self.set_log_file()

        self.set_start_time(time.time())

    def set_log_file(self, filename="log.txt"):
        os.makedirs(self._save_path, mode=0o777, exist_ok=True)
        self.log_file_path = os.path.join(self._save_path, filename)
        with open(self.log_file_path, "w") as fp:
            fp.write("Start Recording!")
        self.stdout(collect_env_info())

    @rank_zero_only
    def stdout(self, outstr):
        with open(self.log_file_path, "a") as fp:
            fp.write(outstr+"\n")
        print(outstr)

    @rank_zero_only
    def set_num_total_steps(self, steps):
        self.num_total_steps = steps

    @rank_zero_only
    def set_start_time(self, tm):
        self.start_time = tm

    @rank_zero_only
    def log_time(self, current_step, current_epoch, batch_idx, batch_size, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / current_step - 1.0) * time_sofar if current_step > 0 else 0
        print_string = "\nEpoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time elapsed: {} | time left: {}"
        self.stdout(print_string.format(current_epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

class Logger(TensorBoardLogger):
    def __init__(self, *args, **kwargs):
        super(Logger, self).__init__(*args, **kwargs)

        self._filewriter = FileWriter(self.log_dir)

    @property
    def filewriter(self):
        return self._filewriter
