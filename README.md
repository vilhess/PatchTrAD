# PatchTrAD: A Transformer-Based Anomaly Detector Using Patch-Wise Reconstruction Error for Time Series

This repository contains the implementation of **PatchTrAD**, a Transformer-based anomaly detection model that leverages patch-wise reconstruction error for effective time series anomaly detection. The model is compatible with multiple datasets, including both univariate and multivariate time series.

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the Datasets

Download and place the datasets as described in the [Datasets](#datasets) section below.

### 3. Run the Model

#### Run on All Datasets

```bash
chmod +x ./run_models.sh
./run_models.sh
```

#### Run on a Specific Dataset

```bash
python main.py dataset=<dataset_name>
```

Replace `<dataset_name>` with one of the following:

- `nyc_taxi`
- `ec2_request_latency_system_failure`
- `smd`
- `smap`
- `msl`
- `swat`

> 📁 **Note:** Dataset-specific hyperparameters are located in the configuration folder: `conf/dataset`.

After training, results are saved to:  
```bash
results/results.json
```

---

## 📊 Datasets

### 🔹 Univariate

- **NYC Taxi Demand**  
  Location: `data/nab`

- **EC2 Request Latency (System Failure)**  
  Location: `data/nab`

Both datasets come from the [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB/).

---

### 🔸 Multivariate

- **SWAT (Secure Water Treatment Testbed)**  
  Location: `data/swat`  
  Source: [iTrust, SUTD](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)

  ⚠ **Note:** Preprocessing is required. Use the functions in:  
  `dataset/swat`

- **Server Machine Dataset (SMD)**  
  Location: `data/smd`  
  Source: [OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly)

  ⚠ **Note:** Preprocessing is required. Use the functions in:  
  `dataset/smd`

- **SMAP & MSL (NASA Telemetry Data)**  
  Location: `data/nasa`  
  Source:  
  [Paper](https://arxiv.org/abs/1802.04431) | [GitHub](https://github.com/khundman/telemanom)

---

### ⚙ Preprocessing

To preprocess SWAT or SMD datasets, run:

```bash
python dataset/preprocess.py
```

---

## 📚 Citation

If you use this project in your research, please cite:

@misc{
      title={PatchTrAD: A Patch-Based Transformer focusing on Patch-Wise Reconstruction Error for Time Series Anomaly Detection}, 
      author={Samy-Melwan Vilhes and Gilles Gasso and Mokhtar Z Alaya},
      year={2025},
      eprint={2504.08827},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.08827}, 
}