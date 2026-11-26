# Hotel cancellations: data and model

A dashboard of the Hotel Booking Demand dataset (a city hotel and a resort hotel in Portugal, 2015–2017)
and the results of the cancellation model from my MLOps project.

- `pipeline/hotel_model.py`, `pipeline/train.py`: the model code (copied from the MLOps project).
- `pipeline/build_data.py`: cleans the data, retrains the final ensemble (Extra Trees V2 + MLP V0,
  averaged, threshold 0.45, chronological split) and writes small JSON files to `data/processed/`.
  It reproduces the notebook's test metrics exactly (ROC AUC 0.804, precision 0.609, recall 0.520).
- `web/`: Next.js site that shows them. Deploy on Vercel with root directory `web`.

## Update the data

```
pip install -r pipeline/requirements.txt
python pipeline/build_data.py
cd web
npm install        # first time only
npm run dev        # http://localhost:3000
```
