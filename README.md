# mnist-visual

## Visual training of neural network to fit MNIST data set
  - Uses feedforward neural network
  - Built on framework from Michael Nielsen's neural networks and deep learning online code
  - Shows training images and the network's response to the image

Imports raw data from pickle, splits data into training, validation, 
and test sets, creates feedforward neural network, trains net on 
mnist images, shows network progress while training

## Dependencies
+ NumPy
  * Data preprocessing and manipulation
+ Matplotlib
  * Data visualization

## Usage
```
python network.py
```

### Optional network tuning in ```main```
+ ```learning rate```
  * default: 0.1
+ ```lmbda```
  * regularization parameter
  * default: 10.0
+ ```epochs```
  * number of epochs to train
  * default: 30
+ ```mini batch size```
  * size of batch to train
  * default: 10
+ network architecture
  * list in network constructor that defines layers and neurons in each
  * default: [784, 100, 10]

### Changelog
+ 0.1.0
  * Initial version