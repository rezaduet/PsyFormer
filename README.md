# PsyFormer: A Multi-Task Ordinal Assessment of Questionnaire-Defined Anxiety and Depression Severity

This repository contains the implementation and experimental pipeline for **PsyFormer**, a stress-Informed multi-task deep learning framework for simultaneous ordinal assessment of anxiety and depression. PsyFormer combines local symptom-pattern extraction using convolutional neural networks (CNNs), global symptom dependency modeling using Transformer encoders, demographic contextual information, gated feature fusion, and multi-task ordinal learning.



<p align="center">
  <img src="https://github.com/rezaduet/PsyFormer-MH/blob/main/architecture%20.png" width="750"/>
</p>


---

## Repository Structure

├── code<br>
├    ├──PsyFormer.ipynb (Proposed Method Code)<br>
├    ├──Baseline_Comparison.ipynb<br>
├    ├──5_Fold_Cross_Val_with_XAI.ipynb<br>
├── dataset <br>
├    ├──Processed.csv<br>
├<br>
├── architecture.png<br>
└── README.md


- To run complete project anybody just need to adjust proper path of the dataset. And then runn all from the Python code from this repository (such as `PsyFormer.ipynb` file), you need to use `Processed.csv` dataset from `dataset` Folder.

---

# How to Run/Reproduce outputs using code(.ipynb) in Google Colab with the given Datasets and codes


## Open Notebook in Google Colab

1. Go to https://colab.research.google.com/
2. Click **File → Upload notebook**
3. Upload the notebook `PsyFormer.ipynb` file  

---

## 📂 Adding Respective Dataset with the Code


### Option 1: Upload Dataset


```python
from google.colab import files
uploaded = files.upload()
```

Then load it:

```python
import pandas as pd
df = pd.read_csv("Processed.csv")
```

### Option 2: Use Google Drive (Recommended)

```python
from google.colab import drive
drive.mount('/content/drive')
```

Authorize access, then load the dataset:

```python
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/path_to_your_dataset/Processed.csv')
```

▶️ Running the Notebook

- Run all cells: Runtime → Run all
- Run individual cells: Shift + Enter

---


This repository contains all results, observations, and inferences displayed in the output (text) cell and the link to the data source.
