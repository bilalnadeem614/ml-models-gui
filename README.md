# ML Models GUI

A simple web app to train, run, and interact with 3 machine learning models through a single Streamlit interface.

**Live App:** https://ml-models-gui.streamlit.app/

## Models Included

| Model | Type | Dataset | Metric |
|---|---|---|---|
| Logistic Regression | Classification | Iris | Accuracy: 1.00 |
| Decision Tree | Classification | Breast Cancer | Accuracy: 0.95 |
| Linear Regression | Regression | California Housing | R² Score: 0.58 |

## How It Works

1. Select a model from the dropdown.
2. Enter the required input values (defaults are pre-filled).
3. Click **Predict** to see the result instantly.

## Tech Stack

- **Python** — model training (scikit-learn)
- **Streamlit** — GUI and deployment
- **joblib** — model saving/loading

## Project Files

- `app.py` — Streamlit application (GUI)
- `lr.py` — trains the Logistic Regression model
- `decision_tree.py` — trains the Decision Tree model
- `linear.py` — trains the Linear Regression model
- `logistic_model.pkl` — saved Logistic Regression model
- `decisiontree_model.pkl` — saved Decision Tree model
- `linear_model.pkl` — saved Linear Regression model
- `requirements.txt` — dependencies

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
