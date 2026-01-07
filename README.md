# Poverty Prediction Challenge

This repository contains the solution for the Poverty Prediction Challenge. The goal is to predict household poverty rates and consumption based on survey data.

## Project Structure

- `data/`: Contains all data files (train, test, submission format).
- `src/`: Source code.
  - `config.py`: Configuration (paths, constants).
  - `data/`: Data loading and management.
  - `features/`: Feature engineering and preprocessing.
  - `models/`: Model definition and training.
  - `utils/`: Helper scripts (submission generation).
- `models/`: Trained models (created after training).
- `main.py`: Main CLI entry point.
- `requirements.txt`: Python dependencies.

## Setup

1.  **Create and Activate Virtual Environment**:

    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

2.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Baseline Model

To train the Random Forest baseline:

```bash
python main.py train
```

This will:
- Load data from `data/`.
- Train a model.
- Save the model to `models/baseline.pkl`.

### Generating Predictions

To generate the submission file:

```bash
python main.py predict
```

This will:
- Load the trained model.
- Predict consumption for the test set.
- Calculate derived poverty rates.
- Save two CSVs to `data/submission/`:
  - `predicted_household_consumption.csv`
  - `predicted_poverty_distribution.csv`

## Notes

- The data must be placed in the `data/` directory.
- The `src/config.py` file controls file paths and column names.