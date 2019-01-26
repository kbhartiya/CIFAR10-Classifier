# CIFAR10-Classifier
- This project is the implementation of the following [paper](http://deeplearning.net/wp-content/uploads/2013/03/dlsvm.pdf).
- Softmax as well as SVM classifier at the end of the Convolutional Neural Network is implemented and compared.
- Download the Cifar 10 batches and save it in the [Cifar_Batches](https://github.com/kbhartiya/CIFAR10-Classifier/tree/master/CIFAR_Batches) folder from the following [link](https://www.cs.toronto.edu/~kriz/cifar.html).
- It is advised to run it on a system with 8GB or above RAM.

To Run Softmax Classifier
-------------------------
Run

```
python3 model.py [-m] --model --Softmax [-e] --epochs [-b] --batch_size
```
To Run SVM Classifier
----------------------
Run

```
python3 model.py [-m] --model --SVM [-e] --epochs [-b] --batch_size
```
Results
--------
- The following results have been obtained by running it on a CPU system with 4GB RAM.


