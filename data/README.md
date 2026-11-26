# Data

- The placeholder dataset (scikit-learn breast cancer) is built in, so nothing needs to be here yet.
- When you pick your real dataset, put the CSV in `data/raw/` and set `data.source: csv` and
  `data.path: data/raw/<file>.csv` in `config.yaml`. If the file is large, commit a download
  script instead and add the CSV to `.gitignore`.
