\# SCDNet: Sparse Cluster Decomposition for Long-term Time Series Forecasting



Official PyTorch implementation of \*\*SCDNet\*\*, a novel long-term time series forecasting model that integrates sparse cluster decomposition with inverted Transformer architecture. SCDNet disentangles common and residual components in time series embeddings, improving long-horizon prediction accuracy on challenging benchmarks (Electricity, Traffic, Weather, ETT, etc.).



\## 📢 Updates

\- \[2024-XX-XX] Code released. Supports multiple datasets and forecasting horizons.



\## 📦 Key Features

\- \*\*Sparse Cluster Decomposition (SCD)\*\*: Learns a dictionary of basis patterns and enforces sparse, orthogonal decomposition to separate shared trends from residuals.

\- \*\*Inverted Transformer Backbone\*\*: Treats variates as tokens, enabling efficient cross-variate attention.

\- \*\*RevIN + Cycle-aware Embeddings\*\*: Reversible instance normalization and cyclic time embeddings to handle distribution shifts and seasonality.

\- \*\*Plug-and-play\*\*: Can be adapted to other Transformer variants (iTransformer, Informer, etc.).



\## 📂 Repository Structure

├── run.py # Main entry for training/testing

├── exp\_long\_term\_forecasting.py # Experiment pipeline

├── data\_provider/ # Data loading and preprocessing

│ ├── data\_factory.py

│ └── data\_loader.py

├── layers/ # Core neural modules

│ ├── SCDNet.py

│ ├── SparseClusterDecomposition.py

│ ├── RevIN.py

│ ├── Embed.py

│ ├── SelfAttention\_Family.py

│ └── Transformer\_EncDec.py

├── utils/ # Metrics, time features, tools

├── scripts/ # Example running scripts

│ ├── SCDNet\_ETTh1.sh

│ ├── SCDNet\_ETTm1.sh

│ ├── SCDNet\_electricity.sh

│ ├── SCDNet\_traffic.sh

│ └── SCDNet\_weather.sh

├── requirements.txt

└── README.md



\## ⚙️ Requirements

\- Python 3.8+

\- PyTorch 2.0.0

\- pandas, numpy, scikit-learn, matplotlib

\- reformer-pytorch (for baseline compatibility)



Install dependencies:

```bash

pip install -r requirements.txt



📊 Data Preparation

Download datasets and place them under ./dataset/:



ETT (ETTh1, ETTh2, ETTm1, ETTm2): Google Drive



Electricity: UCI



Traffic: UCI



Weather: Wetterstation



Ensure the CSV files have a date column and target variable(s). For custom datasets, follow the format in data\_loader.py.



🚀 Training \& Evaluation

Basic Command

python -u run.py \\

&#x20; --is\_training 1 \\

&#x20; --model SCDNet \\

&#x20; --data custom \\

&#x20; --root\_path ./dataset/electricity/ \\

&#x20; --data\_path electricity.csv \\

&#x20; --seq\_len 96 \\

&#x20; --pred\_len 96 \\

&#x20; --enc\_in 321 \\

&#x20; --d\_model 512 \\

&#x20; --n\_heads 4 \\

&#x20; --e\_layers 1 \\

&#x20; --batch\_size 16 \\

&#x20; --cycle 168 \\

&#x20; --n\_clusters 32 \\

&#x20; --top\_k 4 \\

&#x20; --learning\_rate 0.0005 \\

&#x20; --itr 1





Run Scripts for Reproducibility

We provide shell scripts in scripts/ for all benchmark settings. Example:



bash scripts/SCDNet\_electricity.sh



These scripts automatically loop over different prediction lengths (96, 192, 336, 720).



Key Hyperparameters

Parameter	Description

\--n\_clusters	Number of dictionary atoms in SCD module

\--top\_k	Sparsity level (top-k activations)

\--d\_scd	Internal dimension of SCD (default 64)

\--cycle	Cycle length for periodic embeddings (e.g., 24 for hourly, 168 for weekly)

\--use\_norm	Enable RevIN (recommended)

📈 Results

SCDNet achieves state-of-the-art performance on multiple long-term forecasting benchmarks. Detailed results will be provided in the paper. Example test output:



text

mse:0.361, mae:0.xxx

Metrics are saved in results/ and test\_results/.



🧪 Testing Pretrained Model

bash

python -u run.py \\

&#x20; --is\_training 0 \\

&#x20; --model SCDNet \\

&#x20; --checkpoints ./checkpoints/your\_setting/ \\

&#x20; ... (same data args)

📝 Citation



Table\~1 presents the long-term forecasting results. For all datasets, the input length is fixed to 96, and the prediction length is selected from ${96, 192, 336, 720}$. The best and second-best results are highlighted in red and blue, respectively.











🙏 Acknowledgements

We thank the authors of iTransformer, RevIN, and Informer for their open-source contributions.



📧 Contact

For questions or collaboration, please open an issue or contact: zeg@xju.edu.cn]



text















