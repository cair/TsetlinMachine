# The Tsetlin Machine
The code and datasets for the Tsetlin Machine.

Implements the Tsetlin Machine from https://arxiv.org/abs/1804.01508 as well as the multiclass version.

## Requirements

- Python 2.7.x https://www.python.org/downloads/
- Numpy http://www.numpy.org/
- Cython http://cython.org/

## Other Implementations

* Fast bit-operation based implementation in C with MNIST demo, https://github.com/cair/fast-tsetlin-machine-with-mnist-demo
* CUDA implementation with IMDB text classification demo, https://github.com/cair/TextUnderstandingTsetlinMachine
* C implementation, https://github.com/cair/TsetlinMachineC
* C++ Toolkit with Python bindings, https://github.com/WojciechMigda/TsetlinMachineToolkit
* Rust implementation, https://github.com/KhaledSharif/TsetlinMachine
* C++ implementation, https://github.com/222464/TsetlinMachine
* Python framework for rapid deployment, https://github.com/cair/open-tsetlin-machine

# Learning Behaviour
The below figure depicts the learning progress of a Tsetlin Machine on a binarized, but otherwise unenhanced, version of the MNIST dataset (https://en.wikipedia.org/wiki/MNIST_database). See also https://github.com/cair/fast-tsetlin-machine-with-mnist-demo.
![Figure 4](https://github.com/olegranmo/blob/blob/master/learning_progress.png)
As seen in the figure, both test and training accuracy increases steadily epoch-by-epoch. The most notable observation is that while the accuracy on the training data approaches $99.8$\%, no significant overfitting occurs. Accuracy on the test data increases steadily too, approaching an accuracy of $82.2$\% after 250 epochs. This is quite different from what occurs with backpropagation on a neural network, where accuracy on test data starts to drop at some point, without proper regularization mechanisms.

## Licence

Copyright (c) 2018 Ole-Christoffer Granmo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
