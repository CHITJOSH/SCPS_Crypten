**CrypTen**
CrypTen is a machine learning framework built on PyTorch that enables to easily study and develop machine learning models using secure computing techniques.

**Key Contributions of Crypten:**
It is machine learning first. The framework presents the protocols via a CrypTensor object that looks and feels exactly like a PyTorch tensor. This allows the user to use automatic differentiation and neural network modules akin to those in PyTorch. This helps make secure protocols accessible to anyone who has worked in PyTorch.
CrypTen is library-based. Unlike other software in this space, we are not implementing a compiler but instead implement a tensor library just as PyTorch does. This makes it easier for people to debug, experiment, and explore ML models.
The framework is built with real-world challenges in mind. CrypTen does not scale back or oversimplify the implementation of the secure protocols. Parties run in separate processes that communicate with one another. The parties can run on separate machines as well. 


Improvement 1:

CrypTen currently implements a cryptographic method called secure multiparty computation (MPC), with plans to add support for homomorphic encryption and secure enclaves in future releases.

1. Created a class MPCTensor that represents a tensor encrypted using the Paillier homomorphic encryption scheme. 
2. Class method that generates a Paillier key pair (public key and private key) with the specified key length. Returns the generated key pair.
3. An encryption method that encrypts a given integer value using the Paillier public key and returns the encrypted number.
4. Decryption method that decrypts an encrypted number using the Paillier private key and returns the decrypted integer value.
5. Add method: this overloads the addition operator + to perform the addition of two MPCTensor instances. It checks if both operands have the same public key and returns a new MPCTensor instance with the same public key.

Improvement 2:

Currently, performing operations like torch_tensor.add(cryptensor) or torch_tensor + cryptensor is not feasible. The issue arises because functions like __radd__ are not invoked since torch.Tensor.add raises a TypeError instead of a NotImplementedError. This prevents the reverse function from being called.

This addresses the issue with the add, sub, and mul functions. Here's the general approach:
1. Handling Torch Functions: We handle torch.Tensor.{add,sub,mul} functions in the __torch_function__ handler using an @implements decorator.
2. Inheritance: We ensure that subclasses of CrypTensor inherit these decorators by adding an __init_subclass__ function in CrypTensor.
3. Manual Registration: Since MPCTensor dynamically adds functions like add, sub, and mul after the subclass is created, we manually register these functions in MPCTensor.
4. Adjustments: We adjust MPCTensor.binary_wrapper_function to accommodate the specific structure of MPCTensor that torch.Tensor lacks. This involves swapping the order of arguments if necessary and altering the function name to __radd__, __rsub__, etc.


<p align="center"><img width="70%" src="https://raw.githubusercontent.com/facebookresearch/CrypTen/master/docs/_static/img/CrypTen_Identity_Horizontal_Lockup_01_FullColor.png" alt="CrypTen logo" /></p>

[![Support Ukraine](https://img.shields.io/badge/Support-Ukraine-FFD500?style=flat&labelColor=005BBB)](https://opensource.fb.com/support-ukraine) [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/facebookresearch/CrypTen/blob/master/LICENSE) [![CircleCI](https://circleci.com/gh/facebookresearch/CrypTen.svg?style=shield)](https://circleci.com/gh/facebookresearch/CrypTen/tree/master) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/facebookresearch/CrypTen/blob/master/CONTRIBUTING.md)

--------------------------------------------------------------------------------

CrypTen is a framework for Privacy Preserving Machine Learning built on PyTorch.
Its goal is to make secure computing techniques accessible to Machine Learning practitioners.
It currently implements [Secure Multiparty Computation](https://en.wikipedia.org/wiki/Secure_multi-party_computation)
as its secure computing backend and offers three main benefits to ML researchers:

1. It is machine learning first. The framework presents the protocols via a `CrypTensor`
   object that looks and feels exactly like a PyTorch `Tensor`. This allows the user to use
   automatic differentiation and neural network modules akin to those in PyTorch.

2. CrypTen is library-based. It implements a tensor library just as PyTorch does.
   This makes it easier for practitioners to debug, experiment on, and explore ML models.

3. The framework is built with real-world challenges in mind. CrypTen does not scale back or
   oversimplify the implementation of the secure protocols.

Here is a bit of CrypTen code that encrypts and decrypts tensors and adds them

```python
import torch
import crypten

crypten.init()

x = torch.tensor([1.0, 2.0, 3.0])
x_enc = crypten.cryptensor(x) # encrypt

x_dec = x_enc.get_plain_text() # decrypt

y_enc = crypten.cryptensor([2.0, 3.0, 4.0])
sum_xy = x_enc + y_enc # add encrypted tensors
sum_xy_dec = sum_xy.get_plain_text() # decrypt sum
```

It is currently not production ready and its main use is as a research framework.

## Installing CrypTen

CrypTen currently runs on Linux and Mac with Python 3.7.
We also support computation on GPUs.
Windows **is not** supported.

_For Linux or Mac_
```bash
pip install crypten
```

If you want to run the examples in the `examples` directory, you should also do the following
```bash
pip install -r requirements.examples.txt
```

## Examples
To run the examples in the `examples` directory, you additionally need to clone the repo and

```bash
pip install -r requirements.examples.txt
```

We provide examples covering a range of models in the `examples` directory

1. The linear SVM example, `mpc_linear_svm`, generates random data and trains a
  SVM classifier on encrypted data.
2. The LeNet example, `mpc_cifar`, trains an adaptation of LeNet on CIFAR in
  cleartext and encrypts the model and data for inference.
3. The TFE benchmark example, `tfe_benchmarks`, trains three different network
  architectures on MNIST in cleartext, and encrypts the trained model and data
  for inference.
4. The bandits example, `bandits`, trains a contextual bandits model on
  encrypted data (MNIST).
5. The imagenet example, `mpc_imagenet`, performs inference on pretrained
  models from `torchvision`.

For examples that train in cleartext, we also provide pre-trained models in
cleartext in the `model` subdirectory of each example subdirectory.

You can check all example specific command line options by doing the following;
shown here for `tfe_benchmarks`:

```bash
python examples/tfe_benchmarks/launcher.py --help
```

## How CrypTen works

We have a set of tutorials in the `tutorials` directory to show how
CrypTen works. These are presented as Jupyter notebooks so please install
the following in your conda environment

```bash
conda install ipython jupyter
pip install -r requirements.examples.txt
```

1. `Introduction.ipynb` - an introduction to Secure Multiparty Compute; CrypTen's
   underlying secure computing protocol; use cases we are trying to solve and the
   threat model we assume.
2. `Tutorial_1_Basics_of_CrypTen_Tensors.ipynb` - introduces `CrypTensor`, CrypTen's
   encrypted tensor object, and shows how to use it to do various operations on
   this object.
3. `Tutorial_2_Inside_CrypTensors.ipynb` – delves deeper into `CrypTensor` to show
   the inner workings; specifically how `CrypTensor` uses `MPCTensor` for its
   backend and the two different kind of _sharings_, arithmetic and binary, are
   used for two different kind of functions. It also shows CrypTen's
   [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface)-inspired
   programming model.
4. `Tutorial_3_Introduction_to_Access_Control.ipynb` - shows how to train a linear
   model using CrypTen and shows various scenarios of data labeling, feature
   aggregation, dataset augmentation and model hiding where this is applicable.
5. `Tutorial_4_Classification_with_Encrypted_Neural_Networks.ipynb` – shows how
   CrypTen can load a pre-trained PyTorch model, encrypt it and then do inference
   on encrypted data.
6. `Tutorial_5_Under_the_hood_of_Encrypted_Networks.ipynb` - examines how CrypTen
   loads PyTorch models, how they are encrypted and how data moves through a multilayer
   network.
7. `Tutorial_6_CrypTen_on_AWS_instances.ipynb` - shows how to use `scrips/aws_launcher.py`
   to launch our examples on AWS. It can also work with your code written in CrypTen.
8. `Tutorial_7_Training_an_Encrypted_Neural_Network.ipynb` - introduces the
   automatic differentiation functionality of `CrypTensor`. This functionality
   makes it easy to train neural networks in CrypTen.

