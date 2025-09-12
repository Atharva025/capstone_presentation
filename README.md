5G Adversarial Intrusion Detection System (IDS)
1. Problem Statement
Standard signature-based Intrusion Detection Systems are ill-equipped to detect novel Denial of Service (DoS) attacks targeting the 5G Packet Forwarding Control Protocol (PFCP) control plane, creating a significant risk of service disruption. This project will develop a deep learning model to accurately classify malicious versus benign PFCP traffic flows in real-time.

2. Project Objective
To build and evaluate a robust, deep-learning-based Intrusion Detection System for 5G PFCP traffic. The primary goal is to achieve the highest possible F1-Score, balancing the critical need to detect attacks (Recall) while minimizing false alarms that could disrupt network services (Precision).

This project is segmented into two phases:

Baseline Model: Train a neural network on clean, preprocessed data to establish a performance benchmark.

Robust Model (Future Work): Generate adversarial attacks against the baseline model and use adversarial training techniques to build a hardened, robust successor.

3. Setup & Installation
Clone the repository and install the required dependencies.

git clone <your-repo-url>
cd 5G-IDS-Project
pip install -r requirements.txt

4. Usage
The project pipeline is executed via scripts in the src/ directory.

Step 1: Preprocess the Data

Place your raw PFCP_data.csv file into the data/raw/ directory. Then run the preprocessing script:

python src/preprocessing.py --raw_data_path data/raw/PFCP_data.csv --processed_data_dir data/processed

Step 2: Train the Baseline Model

Once the data is processed, run the baseline training script:

# This command will be finalized once baseline.py is written.
python src/baseline.py --data_dir data/processed --model_dir models --results_dir results/baseline
