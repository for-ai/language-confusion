# Language confusion datasets and metrics
Repository for the "Understanding and Mitigating Language Confusion in LLMs" paper


## Computing the Word Pass Rate and Line Pass Rate metrics
```bash
git clone https://github.com/for-ai/language-confusion.git
cd language-confusion
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# Compute and print WPR and LPR for Command R+
python compute_metrics.py outputs/command-r-plus.csv
```