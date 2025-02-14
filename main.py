import torch 
import lightning as L 
from sklearn.metrics import roc_auc_score
import numpy as np
import argparse

from patchtrad import PatchTrad, PatchTradLit
from utils import load_config, save_results

torch.manual_seed(0)

parser = argparse.ArgumentParser(description="Train PatchTrAD on a specified dataset")
parser.add_argument("--dataset", type=str, default="smd", help="Specify dataset")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = args.dataset
av_datasets = ["nyc_taxi", "smd", "smap", "msl", "swat", "ec2_request_latency_system_failure"]
assert dataset in av_datasets, f'Dataset ({dataset}) should be in {av_datasets}'

config = load_config(filename="jsons/config.json", dataset=dataset)

# Some datasets (MSL, SMAP, SMD) are composed of many subsets: score = mean accuracy on each subset, that is why we use list of loaders

if dataset in ["ec2_request_latency_system_failure", "nyc_taxi"]:
    from dataset.nab import get_loaders
    loaders = [get_loaders(window_size=config['ws'], root_dir="data/nab", dataset=dataset, batch_size=config['bs'])]
    print(f"Working on {dataset.upper()}, number of subset: {len(loaders)}")

if dataset in ["smap", "msl"]:
    from dataset.nasa import get_loaders, smapfiles, mslfiles

    if dataset=="smap": file = smapfiles
    elif dataset=="msl": file = mslfiles

    loaders = [get_loaders(window_size=config['ws'], root_dir="data/nasa", dataset=dataset,filename=f, batch_size=config['bs']) for f in file]
    print(f"Working on {dataset.upper()}, number of subset: {len(loaders)}")

if dataset=="smd":
    from dataset.smd import get_loaders, machines
    loaders = [get_loaders(window_size=config['ws'], root_dir="data/smd/processed", machine=m, batch_size=config['bs']) for m in machines]
    print(f"Working on {dataset.upper()}, number of subset: {len(loaders)}")

if dataset=="swat":
    from dataset.swat import get_loaders
    loaders = [get_loaders(window_size=config['ws'], root_dir="data/swat", batch_size=config['bs'])]
    print(f"Working on {dataset.upper()}, number of subset: {len(loaders)}")

aucs = []

for i, (trainloader, testloader) in enumerate(loaders):

    print(f"currently working on subset {i+1}/{len(loaders)}")

    model = PatchTrad(window_size=config["ws"]+1, n_vars=config["in_dim"], stride=config['stride'], patch_len=config['patch_len'], 
                        d_model=config['d_model'], n_heads=config["n_heads"], n_layers=config['n_layers'], d_ff=config["d_ff"], normalize=1, learn_pe=False)

    LitModel = PatchTradLit(model=model)

    trainer = L.Trainer(max_epochs=config["epochs"], logger=False, enable_checkpointing=False)
    trainer.fit(model=LitModel, train_dataloaders=trainloader)

    test_errors = []
    test_labels = []

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

auc = np.mean(aucs)
print(f"Final AUC: {auc}")
save_results(filename="jsons/results.json", dataset=dataset, model=f"patchtrad_S{config['stride']}_L{config['patch_len']}", auc=round(auc, 4))