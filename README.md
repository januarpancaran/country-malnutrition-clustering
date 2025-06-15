# Country Malnutrition Clustering

Clustering Malnutrition in Countries using K-Means and DBHC(DBSCAN-based Hierarchical Clustering) Algorithms

## Dataset

The dataset is obtained from [here](https://www.kaggle.com/datasets/ruchi798/malnutrition-across-the-globe)

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

# Or if you are on a windows machine
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Or you can just access the app by opening [this](https://country-malnutrition-clustering.streamlit.app/)
