# PatchTrAD: A Transformer Patching-Based model focusing on Patch-Wise Reconstruciton error for Time Series Anomaly Detection

In this repository, we implement **PatchTrAD** a Patching-Based Trasformer Anomaly Detector focusing on reconstruction error. The current implementation is designed for several datasets, ranging from univariate to multivariate time series.

---

## Run the Model

To run the model on a specific dataset, use the following command:

```bash
python main.py --dataset <dataset_name>
```
where `<dataset_name>` can be one of the following:  

- `nyc_taxi`  
- `ec2_request_latency_system_failure`  
- `smd`  
- `smap`  
- `msl`  
- `swat`  


Hyperparameters are defined in the configuration file: ``` jsons/config.json```

After each run, the results are saved in: ``` jsons/results.json```

## Datasets

### Univariate

- **NYC Taxi Demand Dataset** (located in the `exps/nyctaxi` folder)  
- **EC2 Request Dataset** (located in the `exps/ec2` folder)  

These two datasets are sourced from the Numenta Anomaly Benchmark (NAB) and can be accessed [here](https://github.com/numenta/NAB/).

### Multivariate

- **SWAT Dataset** (located in the `exps/swat` folder)  

The dataset is provided by iTrust, Centre for Research in Cyber Security, Singapore University of Technology and Design. More details and access requests can be made [here](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/).

- **Server Machine Dataset** (located in the `exps/smd` folder)

This dataset originates from the **OmniAnomaly** methods and can be downloaded by cloning the original [OmniAnomaly repository](https://github.com/NetManAIOps/OmniAnomaly).

- **SMAP Dataset** (located in the `exps/smap` folder)  
- **MSL Dataset** (located in the `exps/msl` folder)  

These datasets contains expert-labeled telemetry anomaly data from the Soil Moisture Active Passive (SMAP) satellite and the Mars Science Laboratory (MSL). They are provided from the **NASA Jet Propulsion Laboratory** [Paper](https://arxiv.org/abs/1802.04431) [GitHub](https://github.com/khundman/telemanom)

---