# Environment
We used Python 3.7 for our experiments. The environment can be set up by following three steps:
1. Install pipenv using pip
2. Install required packages 
3. Activate virtual environment

You can run following commands to set up the environment:
```
pip install pipenv             # To install pipenv if you don't have it already
pipenv install --skip-lock     # To install required packages. 
pipenv shell                   # To activate virtual env
```

If the second step results in issues, you can install packages in Pipfile individually by using pip i.e. "pip install package_name". 

# Data

Data can be downloaded from the url below. Please unzip the data folder, and place it at the root (i.e. at the same level as train.py file).

https://github.com/subtab/subtab/releases/download/0.0/data.zip


# Training & Evaluation
You can train the model using:
```
python train.py 
```
This will also run evaluation at the end of the training. You can also run evaluation separately by using:
```
python eval.py 
```
