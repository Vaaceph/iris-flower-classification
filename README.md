# Iris Flower Classification
[![PEP8](https://img.shields.io/badge/code%20style-pep8-green.svg)](https://www.python.org/dev/peps/pep-0008/)
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](https://github.com/Vaaceph/iris-flower-classification/blob/master/LICENSE.md)  </br></br>
Iris flower classification using Machine learning, also referred as *Hello World* for Machine Learning. It is very basic classification problem which helps understand basic concept of Machine Learning for beginners. </br></br>
Three class for classification are as follows:</br>

* Iris-setosa
* Iris-versicolor
* Iris-virginica

Implementation in [tensorflow](https://github.com/tensorflow/tensorflow) </br> Implementation in [scikit-learn](https://github.com/scikit-learn/scikit-learn) will be released soon.

## Requirement

To install this package, [python3](https://www.python.org/), [pip](https://pypi.org/project/pip/), and [virtual environment](https://docs.python.org/3/library/venv.html) are required.

For Windows users: </br>
[Install python3 and pip](https://phoenixnap.com/kb/how-to-install-python-3-windows)</br>
[Install virtual environment](https://programwithus.com/learn-to-code/Pip-and-virtualenv-on-Windows/)

For Linux users:

```
sudo apt-get update
sudo apt-get install python python3-pip
sudo pip3 install virtualenv 
```

## Installation

Navigate to the cloned directory.

```bash
cd iris-flower-classification
```

Create [virtual environment](https://docs.python.org/3/tutorial/venv.html) for Python.

```bash
python -m venv tutorial-env
```

For Linux users:

```bash
python3 -m venv tutorial-env
```

Here, **tutorial-env** is the name of the virtual environment, you can name it as you like.

Activate virtual environment by the following command: </br>
To activate on windows:

```bash
Scripts\activate
```

On Linux based on OS:

```bash
. tutorial-env/bin/activate
```

Install pip packages from requirements.txt

```bash
pip install -r requirements.txt
```

## Run

On Windows run:

```bash
python classification.py
```

On Linux run:

```bash
python3 classification.py
```

## Data Set

The dataset is downloaded from [Iris Data Set.](http://archive.ics.uci.edu/ml/datasets/Iris)

## Model

### Tensorflow

This classification model is build using deep neural network (DNN) layers. Architecture of DNN is as follows: </br>

* Input layer also known as Layer 0. Size: N * 4

* First deep learning layer, Size: 4 * 20

* Second deep learning layer, Size: 20 * 20

* Output layer, Size: 20 * 3 (Output is 3 because our classification problem has three classes)

### Scikit-learn

Coming soon. Please wait!!

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://github.com/Vaaceph/iris-flower-classification/blob/master/LICENSE.md)
