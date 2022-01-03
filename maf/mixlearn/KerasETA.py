import datetime
import math
from typing import List, Optional

from keras.callbacks import Callback


class KerasETA(Callback):
    def __init__(self, interval: int, epochs: int, pre_text: str = ""):
        super().__init__()
        self.interval: int = interval
        self.epochs: int = epochs
        self.pre_text: str = pre_text
        self.interval_start: Optional[datetime.datetime] = None
        self.keys: List[str] = ['loss', 'val_loss', 'accuracy', 'val_accuracy']

    def on_train_begin(self, logs=None):
        self.interval_start = datetime.datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        if self.interval_start is None:
            self.interval_start = datetime.datetime.now()
        if (epoch % self.interval == 0 and epoch > 0) or self.epochs - epoch == 1:
            now = datetime.datetime.now()
            took_s = (now - self.interval_start).total_seconds()

            left_seconds = (self.epochs - epoch) / self.interval * took_s
            left_minutes = math.floor(left_seconds / 60)
            left_seconds = int(left_seconds % 60)
            self.interval_start = now
            additional = ",".join([f" {k} = {logs[k]}" for k in self.keys])
            pre_text = ""
            if self.pre_text and len(self.pre_text) > 0:
                pre_text = f"{self.pre_text} "
            print("{}ETA {:02d}m {:02d}s for {} epochs left,  v={}ms / {} epochs ... {}".format(pre_text, left_minutes,
                                                                                                left_seconds,
                                                                                                self.epochs - epoch,
                                                                                                took_s,
                                                                                                self.interval,
                                                                                                additional))
            self.interval_start = now
