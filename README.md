# Environment
We used Python 3.7 for our experiments. The environment can be set up by following three steps:
1. Install pipenv using pip
2. Activate virtual environment
3. Install required packages 

You can run following commands to set up the environment:
```
pip install pipenv             # To install pipenv if you don't have it already
pipenv shell                   # To activate virtual env
pipenv install --skip-lock     # To install required packages. 
```

If the last step results in issues, you can install packages in Pipfile individually by using pip i.e. "pip install package_name". 

# Training & Evaluation
You can train the model using:
```
python train.py 
```
This will also run evaluation at the end of the training. You can also run evaluation separately by using:
```
python eval.py 
```
