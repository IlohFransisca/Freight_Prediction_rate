# Freight Rate Prediction (Machine Learning Engineer Assessment)

I predict `posted_rate` for the 12,000 loads in `validation.csv` (dated November and December 2025), using the 48,000 labeled loads in `train-test.csv` (January to October 2025). The notebook copies them into a `data/` folder as `data/validation.csv` and `data/train_test.csv`, the names the instructions and `score.py` use.

**Live app:** https://freightpredictionrate-jkxmrxvei9xl9pgxnnnhau.streamlit.app/ (the code is `app.py` in this repo)
## Submission checklist

| What the instructions ask for | Where it is |
|---|---|
| GitHub repository with solution code, dependencies and run instructions | This repo: `Freight_Prediction_rate.ipynb` (solution), `requirements.txt` (dependencies), this README (run instructions) |
| `validation_predictions.csv` with exactly `load_id,predicted_rate` | `validation_predictions.csv`: 12,000 rows, two columns in that order, built from `validation_predictions_template.csv` |
| Report (PDF or DOCX): approach to the train/validation split | `Freight_Rate_Report.docx`, section 2 |
| Report: fixed December prediction chart produced by `score.py` | `Freight_Rate_Report.docx`, section 6 (`candidate_december.png`) |

## What's in this repo

| File | What it is |
|---|---|
| `Freight_Prediction_rate.ipynb` | The full solution, written for Google Colab |
| `validation_predictions.csv` | My final predictions for the 12,000 validation loads |
| `Freight_Rate_Report.docx` | The report |
| `score.py` | The provided checker, unchanged |
| `requirements.txt` | Python packages for the notebook and the app |
| `train-test.csv`, `validation.csv` | The labeled development data and the 12,000 loads to predict |
| The other input files from the assessment (the template and the December inputs), under the names `score.py` expects |
| `december_chart_inputs_filled.csv` | My predictions for the 31 December rows |
| `candidate_december.png` | The December chart produced by `score.py` |
| `rate_per_mile.png`, `quote_signal_by_month.png` | Figures from the data exploration |
| `app.py` | The Streamlit app (it contains the model code) |
| `freight_rate_model.joblib` | The model saved from the notebook. It only loads on the scikit-learn version it was saved with (1.6.1). The app tries it first and trains the same model itself if it can't be read |

## 1. Setup and how to run

Python 3.10 or newer. Install the packages:

```bash
python -m pip install -r requirements.txt
```

**In Google Colab (how I ran it):**
1. Open `Freight_Prediction_rate.ipynb`.
2. Put the four CSVs and `score.py` in one Drive folder and point `PROJECT_DIR` in the setup cell at it. The notebook copies the CSVs into a `data/` folder there under the names `score.py` expects.
3. Run the cells in order. It takes a few minutes; the validation folds cell is the slow one because it fits three models.

**Locally:** replace the Drive paths in the first cells with local ones and run the cells in order.

The notebook writes `validation_predictions.csv` and `data/december_chart_inputs_filled.csv`, then runs the provided scorer:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs_filled.csv
```

That prints "Validated 12,000 final predictions." and "Validated 31 fixed December predictions." and creates `scorer_results/candidate_december.png`.

## 2. How the notebook is laid out

1. Setup, Drive paths, copying the data into place
2. Loading `train_test.csv`, data-quality checks and basic plots
3. Feature engineering (`FreightFeatureEngineer`)
4. The model (`TwoStageRegressor`, `build_model`)
5. Dropping impossible labels (`drop_label_errors`)
6. Forward-in-time validation, three folds
7. Final fit on all labeled data
8. Predictions for the 12,000 validation loads, filled into the template and saved as `validation_predictions.csv`
9. Diagnostics: the figures and numbers used in the report
10. Optional test of how the model does on cities it never saw (about a minute)
11. December predictions, the scorer, and the December chart

## 3. Key findings from exploring the data

- **Rate per mile** (rate divided by distance) is the useful view. Normal loads sit around 1.7 to 3.2 dollars per mile (median 2.15), and it varies with distance and equipment.
- **`quote_signal` is unreliable.** It follows the rate per mile almost exactly in January to March, June and September (correlation about +0.95), runs the opposite way in April, May, July and October (about -0.98), and is noise in August. In November and December its correlation with my predictions is 0.001 and -0.016, and its daily average barely moves (standard deviation 0.016 against 0.129 in training).
- **The rate level drifts through the year** in a way `market_index` doesn't explain. For example, September had the lowest market index of the year (0.893) but a higher average rate per mile (2.203) than January (2.070 at a market index of 0.927).
- **The validation set has new places.** 736 lanes and 8 cities (Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk, San Diego) never appear in training. Those cities are in 1,447 validation rows, 12.1% of the file.

## 4. Data-quality issues and how I handled them

| Issue | How I handled it |
|---|---|
| 646 training rows with impossible rates (329 below 1 dollar per mile, 317 above 5), separated from normal loads by a clear gap | Dropped from training. Multiplying or dividing them by 5 doesn't put them back near normal (the repaired rows have medians of 3.03 and 1.54 against 2.15), so they can't be repaired. Test data is unaffected |
| Negative weights: 292 in training, 145 in validation | They look like sign errors, so I use the absolute value |
| Missing weight: 300 in training, 165 in validation | Kept as missing with a `weight_missing` flag; the gradient boosting model handles missing values |
| Missing `market_index`: 374 in training, 249 in validation | Filled with that date's average from the same file (features only, no labels) |
| About 1,200 loads at exactly 47,500 lb | Left alone; weights are capped at 47,500 |
| `quote_signal` unstable (section 3) | Not used in the model |
| Unseen lanes and cities (section 3) | The model uses city, coordinates, direction and distance, with no lane encoding. A hold-out test (optional notebook cell) hid 8 cities from training: error on loads touching them was 3.65% and 2.63% in two draws, against about 2.5% for the rest |

## 5. Training and validation approach, and how the data was split

Every validation load is dated after every training load, so the job is forecasting, and a random split would flatter the model. I validated forward in time with an expanding window. Each fold trains on earlier months and tests on the next two:

| Fold | Train | Test | MAE | RMSE | R² | MAPE % |
|---|---|---|---|---|---|---|
| A | Jan-Apr | May-Jun | 68.36 | 165.99 | 0.986 | 2.71 |
| B | Jan-Jun | Jul-Aug | 42.21 | 88.91 | 0.996 | 1.83 |
| C | Jan-Aug | Sep-Oct | 60.71 | 162.43 | 0.986 | 2.54 |
| **Mean** | | | **57.09** | **139.11** | **0.989** | **2.36** |

The final model is refit on all ten months of cleaned data and run on the 12,000 validation loads. The random seed is fixed at 42.

My first version used one 80/20 split, `quote_signal`, and a multiply-by-five repair of the bad labels. It scored MAE 82.83 and MAPE 3.78% on Aug 31 to Oct 31. It isn't strictly comparable with the numbers above (different label handling), but the improvement is clear.

## 6. Reasoning behind the chosen model

A tree model can't carry a slow drift in the rate level into months it hasn't seen, so I split the model in two:

- **Static part:** `HistGradientBoostingRegressor` on log(rate), using distance, weight, equipment, pickup and delivery city, coordinates and direction. It has no date or market columns, so it only learns what depends on the load.
- **Level part:** a linear term for `market_index` (slope about 0.110 on log rate), estimated with 14-day block effects so the drift doesn't bias it, plus one constant: the average leftover level over the last 60 training days (-0.0953), carried forward into November and December.

I chose to carry the latest level forward instead of fitting a time trend, because projecting a trend two months ahead is a bigger assumption than assuming the level stays near where it was.

## 7. Code walkthrough: the most important parts

| Where | What it does |
|---|---|
| `drop_label_errors` | Removes training rows with rate per mile outside 1 to 5 |
| `FreightFeatureEngineer.transform` | Cleans weights, fills `market_index`, builds `log_distance` and direction features, and does not use `quote_signal` |
| `TwoStageRegressor.fit` | Alternates between the gradient boosting model and the linear market and level terms, then stores the carried-forward level |
| `TwoStageRegressor.predict` | Combines the trees, the market term and the carried-forward level, and converts back from log to dollars |
| Folds cell | The three forward-in-time folds and the metrics table |
| Predict-validation cell | Runs the final model on every row of `validation.csv`, fills the template, checks the 12,000 rows, and saves `validation_predictions.csv` |
| December cell | Builds the 31 December rows, using each date's average `market_index` from `validation.csv` and city coordinates from training, then runs `score.py` |

## 8. December chart

`score.py` produces `scorer_results/candidate_december.png` from the 31 fixed rows (Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb). Only the date changes, so the weekly rise and fall comes from the market index. Predictions run from $827.59 to $847.36 (about 2.30 to 2.35 dollars per mile).

## 9. The web app

The app lets you try the model on any load, run one load across a date range, or score a CSV of loads.

- It tries to load my saved model first. The server runs a newer scikit-learn than my notebook did (1.9.1 against 1.6.1), so it can't read that file and trains its own copy instead, with the same code, settings and data as my notebook. This takes under a minute, and the About page says which route it took.
- If it has been idle, Streamlit puts it to sleep. Tap the wake-up button and give it a minute.
- To run it yourself: `python -m pip install -r requirements.txt`, then `streamlit run app.py`.
- The app uses each day's average market index, while the notebook uses each load's own value. The difference is about 0.2% on average.

## 10. Limitations

- The training data ends in October, so a real holiday effect in late December can't be learned.
- Carrying the level forward assumes November and December stay near the September and October level.
- I don't know how the final score treats impossible labels; if the validation set has such rows, no model can predict them from the features.
- About 12% of validation loads involve a city the model never saw, so they are probably a little less accurate.
- A `.joblib` model saved from the notebook only loads on the scikit-learn version it was saved with.
