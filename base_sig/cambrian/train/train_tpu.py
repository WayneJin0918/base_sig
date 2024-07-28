
from cambrian.train.train_fsdp import train

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple


if __name__ == "__main__":
    #train()
    import multiprocessing as mp
    import torch_xla.distributed.xla_multiprocessing as xmp
    mp.set_start_method('spawn', force=True)

    import os
    if os.getenv('LLAVA_DEBUG', None) == '1':
        xmp.spawn(train, args=(None,), nprocs=1)
    else:
        xmp.spawn(train, args=(None,))
