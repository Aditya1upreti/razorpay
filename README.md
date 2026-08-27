# Sentinel — AI Risk Manager + Revenue Recovery
### Razorpay AI Buildathon — Track 02 + Track 03

> One pipeline: an AI investigator that traces hidden connections between
> accounts, verifies its own verdict before trusting it, and routes anything
> that isn't fraud straight into automated recovery.

---

## Problem Statement
*(fill in after Day 1 — one paragraph, plain language)*

## Architecture
*(paste the architecture diagram from the build plan here, plus a short
paragraph per layer)*

## Evaluation Results
*(fill in after Day 6 — precision, recall, false-positive rate, calibration
summary, recovery ₹ figures)*

## Defense-Only Statement
This tool produces risk scores, explanations, and merchant-facing outputs
only. It never generates attacker-facing guidance, evasion techniques, or
any output that could help someone commit fraud.

## Privacy-By-Design Statement
All identifiers (card, device, account, IP) are hashed/masked before any
data is sent to an external AI API. The AI reasoning layer never receives
raw card numbers, IP addresses, or personal information — only anonymized
tokens and computed risk signals.

---

## How to Run

### 1. Prerequisites
- Python 3.10 or higher (check with `python3 --version`)
- An Anthropic API key (get one at https://console.anthropic.com/)
- Git

### 2. Clone and set up the environment
```bash
git clone <your-repo-url>
cd sentinel-fraud-detector

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure your API key
```bash
cp .env.example .env
# then open .env and paste your real Anthropic API key
```

### 4. Generate the synthetic dataset
```bash
python data/generate_dataset.py
```

### 5. Run the full pipeline + evaluation
```bash
python evaluate.py
```

### 6. Launch the dashboard
```bash
streamlit run dashboard/app.py
```
This opens a local browser window (usually `http://localhost:8501`).

---

## Repo Structure
```
sentinel-fraud-detector/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   ├── generate_dataset.py
│   ├── transactions_train.csv       (generated)
│   ├── transactions_test.csv        (generated, frozen after Day 1)
│   └── account_relationships.csv    (generated)
├── engine/
│   ├── __init__.py
│   ├── rules.py                     ← Layer 1: deterministic scoring
│   ├── graph_builder.py             ← Layer 2: ring detection + BFS trail
│   ├── masking.py                   ← privacy layer (build before ai_investigator)
│   ├── ai_investigator.py           ← Layer 3: dual-pass AI + override + fallback
│   └── recovery.py                  ← Track 03: soft/hard decline routing
├── evaluate.py                      ← metrics + confidence calibration
├── results/
│   ├── evaluation_report.json       (generated)
│   ├── evaluation_report.md         (generated)
│   └── calibration_table.csv        (generated)
├── dashboard/
│   └── app.py                       ← Streamlit UI
├── docs/
│   └── architecture-diagram.png
└── tests/
    └── test_rules.py
```

## Tech Stack
- **Python 3.10+**
- **pandas / numpy** — data handling
- **networkx** — relationship graph + ring detection
- **anthropic** (Claude API) — AI investigator reasoning layer
- **streamlit + plotly** — dashboard
- **scikit-learn** — evaluation metrics
- **faker** — synthetic data generation
