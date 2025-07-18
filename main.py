import torch
import lightning as L
from lightning.pytorch import seed_everything
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from pytorch_lightning.loggers import WandbLogger
import gc
from datetime import datetime

from patchtrad import PatchTradLit
from utils import save_results

torch.multiprocessing.set_sharing_strategy('file_system')

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"---------")
    print("Config:")
    print(OmegaConf.to_yaml(cfg))
    print(f"---------")

    config = cfg.dataset
    dataset = config.name

    av_datasets = ["nyc_taxi", "smd", "smap", "msl", "swat", "ec2_request_latency_system_failure"]
    assert dataset in av_datasets, f"Dataset ({dataset}) should be in {av_datasets}"

    if dataset in ["ec2_request_latency_system_failure", "nyc_taxi"]:
        from dataset.nab import get_loaders
        loaders = [get_loaders(window_size=config.ws, root_dir="data/nab", dataset=dataset, batch_size=config.bs)]

    elif dataset in ["smap", "msl"]:
        from dataset.nasa import get_loaders, smapfiles, mslfiles
        file = smapfiles if dataset == "smap" else mslfiles
        loaders = [get_loaders(window_size=config.ws, root_dir="data/nasa", dataset=dataset, filename=f, batch_size=config.bs) for f in file]

    elif dataset == "smd":
        from dataset.smd import get_loaders, machines
        loaders = [get_loaders(window_size=config.ws, root_dir="data/smd/processed", machine=m, batch_size=config.bs) for m in machines]

    elif dataset == "swat":
        from dataset.swat import get_loaders 
        loaders = [get_loaders(window_size=config.ws, root_dir="data/swat", batch_size=config.bs)]

    run_name = f"{dataset}_PatchTrAD_{datetime.now().strftime('%m%d_%H%M')}"
    wandb_logger = WandbLogger(project='PatchTrAD', name=run_name)
    
    aucs = []
    
    for i, (trainloader, testloader) in enumerate(loaders):
        seed_everything(0)
        print(f"Currently working on subset {i+1}/{len(loaders)}")
   
        LitModel = PatchTradLit(config)
        
        trainer = L.Trainer(max_epochs=config.epochs, logger=wandb_logger, enable_checkpointing=False, log_every_n_steps=1, accelerator=DEVICE)
        trainer.fit(model=LitModel, train_dataloaders=trainloader)
        
        results = trainer.test(model=LitModel, dataloaders=testloader)
        auc = results[0]["auc"]

        print(f"AUC: {auc}")
        aucs.append(auc)

        wandb_logger.experiment.summary[f"auc_subset_{i+1}/{len(loaders)}"] = auc

        if DEVICE == "cuda": # To empty the gpu after each loop
            LitModel.to("cpu")
            del LitModel
            del trainloader, testloader
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            del trainer
            trainer = None
            gc.collect()
            torch.cuda.empty_cache()
    
    final_auc = np.mean(aucs)
    print(f"Final AUC: {final_auc}")
    #save_results(filename="results/results.json", dataset=dataset, model=f"patchtrad", auc=round(final_auc, 4))

    wandb_logger.experiment.summary["final_auc"] = final_auc
    wandb.finish()

if __name__ == "__main__":
    main()
