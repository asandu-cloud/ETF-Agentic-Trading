## AI Generated Set of instructions to avoid a lot of debugging

## ⚙️ Setup & Usage Instructions

These steps assume Python ≥ 3.9 and git are installed.

1. Clone the repo
git clone https://github.com/<your-username>/DDMI-AgenticTrading.git
cd DDMI-AgenticTrading

2. Create and activate a virtual environment

macOS / Linux

python3 -m venv .venv
source .venv/bin/activate


Windows (PowerShell)

python -m venv .venv
.\.venv\Scripts\Activate.ps1


(You should see (.venv) or similar in your terminal prompt.)

3. Install Python dependencies
pip install -r requirements.txt

4. (Optional but recommended) Register a Jupyter kernel for this env

This makes it easy to use the correct environment in notebooks (VS Code / JupyterLab).

python -m pip install ipykernel
python -m ipykernel install --user --name DDMI-AgenticTrading --display-name "DDMI-AgenticTrading"


Later, in VS Code or Jupyter, select the kernel named:

DDMI-AgenticTrading

for all notebooks in this project.

5. Project structure (key folders)
DDMI-AgenticTrading/
├─ data/
│  ├─ raw/          # downloaded price data, etc.
│  └─ processed/    # features, sentiment, agent signals
├─ src/
│  ├─ agents/       # sentiment_agent.py, technical_agent.py, risk_agent.py
│  ├─ config.py     # tickers, dates, global params
│  ├─ data_loader.py
│  ├─ features.py
│  ├─ backtest.py
│  ├─ coordinator.py
│  └─ portfolio.py
└─ notebooks/
   ├─ 01_download_prices.ipynb
   └─ ...


Note: data_loader.py uses paths relative to the project root, so it will always save to data/raw/prices_yahoo.csv regardless of where you run it from.

6. Running modules from the command line

Always run project code as modules from the project root, not as loose scripts.

From the repo root (DDMI-AgenticTrading):

# Download & save price data for all tickers in config.py
python -m src.data_loader

# (later) compute technical features
python -m src.features

# (later) run backtest
python -m src.backtest


This ensures imports like from src.config import ... work correctly.

7. Using the project from notebooks

Open the .ipynb file inside the notebooks/ folder.

In VS Code / Jupyter, select kernel: DDMI-AgenticTrading.

At the top of each notebook, add:

import os, sys

# Set working directory to project root (one level up from /notebooks)
os.chdir("..")   # if the notebook is in /notebooks

# Make 'src' importable as a package
sys.path.append(".")


Now you can use:

from src.data_loader import download_prices, load_prices

prices = download_prices()
prices.tail()


No manual path hacking beyond this is needed.

8. Common pitfalls (and how we avoid them)

❌ Running scripts by path
python src/data_loader.py

This breaks package imports.

✔ Correct:
python -m src.data_loader

❌ Using the wrong Python environment for notebooks

Leads to “module not found” (yfinance, websockets, etc.)

✔ Fix: select the DDMI-AgenticTrading kernel (created in step 4).

❌ Working directory inside /notebooks without adjusting paths

Leads to “Cannot save file into non-existent directory 'data/raw'”.

✔ Fix: os.chdir("..") at the top of each notebook, or start Jupyter from the project root.