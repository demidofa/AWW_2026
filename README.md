# AWW_2026
AI Detective for Petroleum Data  Core analysis data have been accumulated for decades — from archives, laboratories, and digital systems. Hidden within these datasets are errors, inconsistencies, and anomalies. The goal of this project is to identify them systematically and develop an intelligent tool that can detect such issues automatically.

# AI Detective for Petroleum Data

## Project Context

### Why It Matters

Oil and gas companies build geological models using core analysis data — measurements obtained from rock samples extracted from wells. The quality of these data directly affects reserve estimation, reservoir characterization, and production planning.

### The Problem

Core analysis data have been accumulated over decades from paper archives, laboratory reports, and digital systems. During digitization and integration, errors inevitably appear: incorrect values, missing records, inconsistent measurements, duplicates, and physically impossible combinations.

If these errors are not detected, geological models may produce unreliable results, leading to incorrect engineering and economic decisions.

### Real-World Task

The team receives a “dirty” dataset containing **303 records from 10 wells**. The task is to detect all errors of six predefined types and develop a tool that can automatically verify any CSV file containing core analysis data.

### Team Objective

The main goal is to create an automated data verification tool for petroleum core datasets. The tool should:

- load a CSV file with core analysis data;
- perform exploratory data analysis;
- detect errors and anomalies;
- check physical and statistical consistency of the data;
- generate a clear verification report.

### What Participants Will Gain

By completing this project, participants will gain:

- experience with a real industrial data analysis problem;
- practical skills in exploratory data analysis;
- experience in statistical validation of engineering data;
- a portfolio-ready project case;
- an opportunity to enter a master’s program without entrance examinations.

---

## Dataset

The project uses the dataset:
core_data_dirty_v2.csv

## Dataset Characteristics

The dataset contains:

- **303 rows**
- **10 wells**
- **3 laboratories**
- **well coordinates**
- **geological zones A, B, and C**

---

## Data Description

| Column | Description | Valid Range / Constraint |
|---|---|---|
| `well_id` | Well identifier (`Well_1` — `Well_10`) | — |
| `depth_m` | Depth at which the core sample was taken | meters |
| `porosity` | Porosity: fraction of pore volume in the rock | 0 — 1 |
| `permeability_mD` | Permeability: ability of the rock to transmit fluids | > 0 mD |
| `density_gcc` | Rock density. Typical values: quartz — 2.65, calcite — 2.71, dolomite — 2.87 | 2.0 — 2.9 |
| `water_saturation` | Water saturation: fraction of pore volume filled with water | 0 — 1 |
| `oil_saturation` | Oil saturation: fraction of pore volume filled with oil | `Sw + So ≤ 1` |
| `year` | Year of laboratory analysis | 1990 — 2019 |
| `lab_id` | Laboratory identifier (`Lab_A`, `Lab_B`, `Lab_C`) | — |
| `x_m`, `y_m` | Well coordinates within the oilfield | meters |
| `geol_zone` | Geological zone (`A`, `B`, `C`) | — |

---

## Expected Result

The final result of the project should be a reproducible software tool that automatically checks petroleum core analysis data and identifies potential data quality problems.

The tool should produce:

- a summary of the dataset;
- detected errors grouped by type;
- statistical and physical consistency checks;
- visualizations for exploratory data analysis;
- a final report with recommendations for data correction.

---

## Project Theme

### AI Detective for Petroleum Data

Core analysis data have been accumulated for decades — from archives, laboratories, and digital systems. Hidden within these datasets are errors, inconsistencies, and anomalies.

The goal of this project is to identify them systematically and develop an intelligent tool that can detect such issues automatically.


<img width="1277" height="443" alt="Figure 2026-05-07 185150 (3)" src="https://github.com/user-attachments/assets/2f4af11d-c1e9-4c0c-a665-86f59e8eaa56" />
<img width="857" height="498" alt="Figure 2026-05-07 185150 (6)" src="https://github.com/user-attachments/assets/f2deb86f-6e2e-4a2f-8ce7-b363758272ef" />

