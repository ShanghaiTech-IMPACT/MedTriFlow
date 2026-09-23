import argparse, os, sys
import shutil as sh
import time
import gc

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)

from omegaconf import OmegaConf

from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor, DeviceStatsMonitor
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only

from util import instantiate_from_config, load_config

@rank_zero_only
def rank_zero_print(*args):
    print(*args)

def get_parser(**parser_kwargs):
    """Build the command-line parser used by training and evaluation runs."""
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-r", "--resume", dest='resume', action='store_true', default=False)
    parser.add_argument("-b", "--base", type=str, default='configs/syncdreamer-training.yaml',)
    parser.add_argument("-l", "--logdir", type=str, default="./exp", help="directory for logging data", )
    parser.add_argument("-n", "--name", type=str, default="", help="trial name", )
    parser.add_argument("-s", "--seed", type=int, default=6033, help="seed for seed_everything", )
    parser.add_argument("--finetune_from", type=str, default="", help="path to checkpoint to load model state from" )
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--test", type=str, default="")
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--static_graph", action='store_true', default=False)
    parser.add_argument("--not_find_unused_parameters", action='store_true', default=False)
    return parser

class SetupCallback(Callback):
    """Create experiment directories and snapshot the resolved config and code."""

    def __init__(self, resume, logdir, ckptdir, cfgdir, val_dir, test_dir, code_dir, config):
        super().__init__()
        self.resume = resume
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.code_dir = code_dir

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)
            os.makedirs(self.val_dir, exist_ok=True)
            os.makedirs(self.test_dir, exist_ok=True)
            os.makedirs(self.code_dir, exist_ok=True)
            
            ignore_patterns = sh.ignore_patterns('*.pyc', '*.o', '*.e', '__pycache__', 'tmp')
            sh.copytree(os.getcwd(), self.code_dir, ignore=ignore_patterns, dirs_exist_ok=True)

            rank_zero_print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config, os.path.join(self.cfgdir, "configs.yaml"))

            if not self.resume and os.path.exists(os.path.join(self.logdir,'checkpoints','last.ckpt')):
                raise RuntimeError(f"checkpoint {os.path.join(self.logdir,'checkpoints','last.ckpt')} existing")

class CUDACallback(Callback):
    """Record per-epoch wall time and peak CUDA memory for the active device."""

    def on_train_epoch_start(self, trainer, pl_module):
        if hasattr(trainer.datamodule, "sampler"):
            trainer.datamodule.sampler.set_epoch(trainer.current_epoch)
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        try:
            max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2 ** 20
            epoch_time = time.time() - self.start_time
            max_memory = trainer.strategy.reduce(max_memory)
            epoch_time = trainer.strategy.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


class ResumeCallBacks(Callback):
    """Compatibility callback kept for resumed Lightning runs."""

    def on_train_start(self, trainer, pl_module):
        return

class CleanModelCheckpoint(ModelCheckpoint):
    """Checkpoint callback that releases temporary CUDA and Python memory."""

    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)

        trainer.strategy.barrier()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            
            
def load_model(new_model, finetune_from, strict=True):
    """Load a Lightning checkpoint state dictionary into ``new_model``."""
    rank_zero_print(f"Attempting to load state from {finetune_from}")
    old_state = torch.load(finetune_from, map_location="cpu", weights_only=False)["state_dict"]
    new_model.load_state_dict(old_state, strict=strict)

def get_optional_dict(name, config):
    """Return an optional config section, or an empty OmegaConf container."""
    if name in config:
        cfg = config[name]
    else:
        cfg =  OmegaConf.create()
    return cfg

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    sys.path.append(os.getcwd())
    opt, extras = get_parser().parse_known_args()
    
    assert opt.base != ''
    if opt.name == "":
        name = os.path.split(opt.base)[-1]
        name = os.path.splitext(name)[0]
    else:
        name = opt.name
    logdir = os.path.join(opt.logdir, name)

    ckptdir = os.path.join(logdir, 'ckpts')
    cfgdir = os.path.join(logdir, "configs")
    val_dir = os.path.join(logdir, 'val')
    test_dir = os.path.join(logdir, 'test') if opt.output=="" else opt.output
    code_dir = os.path.join(logdir, 'code')
    
    seed_everything(opt.seed)

    config = load_config(opt.base, cli_args=extras)
    
    lightning_config = config.lightning
    trainer_config = config.lightning.trainer
    if opt.debug:
        accelerator = 'cpu'
        rank_zero_print(f"Running on CPU for debug !!!")
    else:
        accelerator = 'cuda'
        gpuinfo = trainer_config["devices"]
        rank_zero_print(f"Running on GPUs {gpuinfo}")
        ngpu = len(gpuinfo)

    model = instantiate_from_config(config.model)
    model.cpu()
    if opt.finetune_from != "":
        load_model(model, opt.finetune_from)
    
    ckpt_path = None
    if opt.resume:
        ckpt = os.path.join(ckptdir, "last.ckpt")
        ckpt_path = ckpt
        opt.finetune_from = "" # disable finetune checkpoint
    
    if opt.resume_from != "":
        ckpt_path = opt.resume_from

    default_logger_cfg = {"target": "pytorch_lightning.loggers.TensorBoardLogger",
                          "params": {"save_dir": logdir, "name": "tensorboard_logs", }}
    logger_cfg = OmegaConf.create(default_logger_cfg)
    logger = instantiate_from_config(logger_cfg)

    default_modelckpt_cfg = {"target": "run.CleanModelCheckpoint",
                             "params": {"dirpath": ckptdir, "filename": "{epoch:06}", "verbose": True, "save_last": True, "every_n_train_steps": 2000}}
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, get_optional_dict("modelcheckpoint", lightning_config))
    default_modelckpt_cfg_repeat = {"target": "run.CleanModelCheckpoint",
                                     "params": {"dirpath": ckptdir, "filename": "{step:08}", "verbose": True, "save_last": False, "every_n_train_steps": 5000, "save_top_k": -1, "save_weights_only": False}}
    modelckpt_cfg_repeat = OmegaConf.merge(default_modelckpt_cfg_repeat)

    default_callbacks_cfg = {
        "setup_callback": {
            "target": "run.SetupCallback",
            "params": {"resume": opt.resume, "logdir": logdir, "ckptdir": ckptdir, "cfgdir": cfgdir, "val_dir": val_dir, "test_dir": test_dir, "code_dir": code_dir, "config": config}
        },
        "learning_rate_logger": {
            "target": "run.LearningRateMonitor",
            "params": {"logging_interval": "step"}
        },
    }
    
    if not opt.debug:
        default_callbacks_cfg["cuda_callback"] = {"target": "run.CUDACallback"}
    
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, get_optional_dict("callbacks", lightning_config))
    callbacks_cfg['model_ckpt'] = modelckpt_cfg
    callbacks_cfg['model_ckpt_repeat'] = modelckpt_cfg_repeat
    callbacks = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    
    if opt.resume or opt.resume_from != "":
        callbacks.append(ResumeCallBacks())
        
    if opt.debug:
        callbacks.append(DeviceStatsMonitor(cpu_stats=True))
    
    if ngpu>1:
        trainer = Trainer(**trainer_config, accelerator=accelerator, strategy=DDPStrategy(find_unused_parameters=True if not opt.not_find_unused_parameters else False, static_graph=opt.static_graph), 
                          logger=logger, callbacks=callbacks)
    else:
        trainer = Trainer(**trainer_config, accelerator=accelerator, logger=logger, callbacks=callbacks)
    
    trainer.logdir = logdir

    config.data.params.seed = opt.seed
    data = instantiate_from_config(config.data)
    data.prepare_data()

    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    accumulate_grad_batches = trainer_config.accumulate_grad_batches if hasattr(trainer_config, "accumulate_grad_batches") else 1
    rank_zero_print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    model.learning_rate = base_lr
    rank_zero_print("++++ NOT USING LR SCALING ++++")
    rank_zero_print(f"Setting learning rate to {model.learning_rate:.2e}")
    model.val_dir = val_dir
    model.test_dir = test_dir
    model.val_files = data.val_data
    model.min = -0.5
    model.max = 0.5
    model.threshold = 0.5
    if opt.resume or opt.resume_from != "":
        model.resume = True
    else:
        model.resume = False
    if opt.test=="":
        data.setup('fit')        
        trainer.fit(model, data, ckpt_path=ckpt_path)
    else:
        data.setup('test')
        model.test_files = data.test_data
        load_model(model, opt.test)
        trainer.test(model, data)
        
