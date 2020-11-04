# Home Credit Prediction
Home Credit is a lender who provides loans to populations unable to use traditional credit services. Goal is to lower loan risk by identifying patterns from within historical data. Let's figure out the most predictive data points through some machine learning models.

Data used is the Home_Loan dataset.

## Requirements
- condamini or conda
- or pyenv with Python: 3.8.5

## Environment

home_loan includes:

jupyterlab 2.2.9
```bash
conda install -c conda-forge jupyterlab
```

pandas 1.1.3
```bash
conda install pandas
````

numpy 1.19.2
```bash
conda install numpy
````

scikit-learn 0.23.2
```bash
conda install -c anaconda scikit-learn
```
matplotlib 3.3.2
```bash
conda install -c conda-forge matplotlib
````

seaborn 0.11.0
```bash
conda install -c anaconda seaborn
````

mlxtend 0.17.3
```bash
conda install -c conda-forge mlxtend
```

imbalanced-learn 0.7.0
```bash
-c conda-forge imbalanced-learn
```

## Setup
Having Anaconda installed then create your ENV with

```bash
make setup-conda
```

With pyenv installed

```bash
make setup-pyenv
```

## Usage

In order to train the model and store test data in the data folder and the model in models run:

```bash
python train.py  
```

In order to test that predict works on a test set you created run:

```bash
python predict.py models/model.sav Home_Loan/X_test.csv Home_Loan/y_test.csv
```

## Limitations

This project is work in progress.