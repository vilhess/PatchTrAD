import torch
import lightning as L
from sklearn.metrics import roc_auc_score
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from pytorch_lightning.loggers import WandbLogger

from patchtrad import PatchTrad, PatchTradLit
from utils import save_results
from dataset.nab import get_loaders as get_nab_loaders
from dataset.nasa import get_loaders as get_nasa_loaders, smapfiles, mslfiles
from dataset.smd import get_loaders as get_smd_loaders, machines
from dataset.swat import get_loaders as get_swat_loaders

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
        loaders = [get_nab_loaders(window_size=config.ws, root_dir="data/nab", dataset=dataset, batch_size=config.bs)]
    elif dataset in ["smap", "msl"]:
        file = smapfiles if dataset == "smap" else mslfiles
        loaders = [get_nasa_loaders(window_size=config.ws, root_dir="data/nasa", dataset=dataset, filename=f, batch_size=config.bs) for f in file]
    elif dataset == "smd":
        loaders = [get_smd_loaders(window_size=config.ws, root_dir="data/smd/processed", machine=m, batch_size=config.bs) for m in machines]
    elif dataset == "swat":
        loaders = [get_swat_loaders(window_size=config.ws, root_dir="data/swat", batch_size=config.bs)]

    wandb_logger = WandbLogger(project='PatchTrAD', name=f"dataset_{dataset}")
    
    aucs = []
    
    for i, (trainloader, testloader) in enumerate(loaders):
        torch.manual_seed(0)
        print(f"Currently working on subset {i+1}/{len(loaders)}")
        
        model = PatchTrad(window_size=config.ws+1, n_vars=config.in_dim, stride=config.stride, patch_len=config.patch_len,
                          d_model=config.d_model, n_heads=config.n_heads, n_layers=config.n_layers, d_ff=config.d_ff,
                          normalize=1, learn_pe=False)
        
        LitModel = PatchTradLit(model=model)
        trainer = L.Trainer(max_epochs=config.epochs, logger=wandb_logger, enable_checkpointing=False, log_every_n_steps=1)
        trainer.fit(model=LitModel, train_dataloaders=trainloader)
        
        test_errors, test_labels = [], []
        model = LitModel.model.to(DEVICE)
        model.eval()
        
        with torch.no_grad():
            for x, anomaly in testloader:
                x = x.to(DEVICE)
                errors = model.get_loss(x, mode="test")
                test_labels.append(anomaly)
                test_errors.append(errors)
        
        test_errors = torch.cat(test_errors).detach().cpu()
        test_labels = torch.cat(test_labels).detach().cpu()
        test_scores = -test_errors
        
        auc = roc_auc_score(y_true=test_labels, y_score=test_scores)
        print(f"AUC: {auc}")
        aucs.append(auc)

        wandb_logger.experiment.config[f"auc_subset_{i+1}/{len(loaders)}"] = auc
    
    final_auc = np.mean(aucs)
    print(f"Final AUC: {final_auc}")
    save_results(filename="results/results.json", dataset=dataset, model=f"patchtrad", auc=round(final_auc, 4))

    wandb_logger.experiment.config["final_auc"] = final_auc
    wandb.finish()

if __name__ == "__main__":
    main()
