# The Tsetlin Machine

The code and datasets for the Tsetlin Machine. Implements the Tsetlin Machine from https://arxiv.org/abs/1804.01508, including the multiclass version. The Tsetlin Machine solves complex pattern recognition problems with easy-to-interpret propositional formulas, composed by a collective of Tsetlin Automata.

<p align="center">
  <img width="75%" src="https://github.com/olegranmo/blob/blob/master/Tsetlin_Machine_Example_Configuration_Full.png">
</p>

## Basics

<p align="left">
A basic Tsetlin Machine takes a vector <img src="http://latex.codecogs.com/svg.latex?X=[x_1,\ldots,x_o]" border="0" valign="middle"/> of binary features as input, to be classified into one of two classes, <img src="http://latex.codecogs.com/svg.latex?y=0" border="0" valign="middle"/> or <img src="http://latex.codecogs.com/svg.latex?y=1" border="0" valign="middle"/>. Together with their negated counterparts, <img src="http://latex.codecogs.com/svg.latex?\bar{x}_k = {\lnot} {x}_k = 1-x_k" border="0" valign="middle"/>, the features form a literal set <img src="http://latex.codecogs.com/svg.latex?L = \{x_1,\ldots,x_o,\bar{x}_1,\ldots,\bar{x}_o\}" border="0" valign="middle"/>.
</p>

<p align="left">
A Tsetlin Machine pattern is formulated as a conjunctive clause <img src="http://latex.codecogs.com/svg.latex?C_j" border="0" valign="middle"/>, formed by ANDing a subset <img src="http://latex.codecogs.com/svg.latex?L_j {\subseteq} L" border="0" valign="middle"/> of the literal set:
</p>

<p align="center">
  <img src="http://latex.codecogs.com/svg.latex?C_j (X)=\bigwedge_{{{l}_{k}} {\in} L_j} l_k = \prod_{{{l}_{k}} {\in} L_j} l_k" border="0" valign="middle"/>.
</p>

<p align="left">
For example, the clause <img src="http://latex.codecogs.com/svg.latex?C_j(X) = x_1 {\land} x_2 = x_1 x_2" border="0" valign="middle"/> consists of the literals <img src="http://latex.codecogs.com/svg.latex?L_j = \{x_1, x_2\}" border="0" valign="middle"/> and outputs <img src="http://latex.codecogs.com/svg.latex?1" border="0" valign="middle"/> iff <img src="http://latex.codecogs.com/svg.latex?x_1 = x_2 = 1" border="0" valign="middle"/>.
</p>

<p align="left">
The number of clauses employed is a user set parameter <img src="http://latex.codecogs.com/svg.latex?n" border="0"/>. Half of the clauses are assigned positive polarity. The other half is assigned negative polarity. The clause outputs, in turn, are combined into a classification decision through summation and thresholding using the unit step function <img src="http://latex.codecogs.com/svg.latex?u(v) = 1 ~\mathbf{if}~ v \ge 0 ~\mathbf{else}~ 0" border="0" valign="middle"/>:
</p>

<p align="center">
<img src="http://latex.codecogs.com/svg.latex?\hat{y} = u\left(\sum_{j=1}^{n/2} C_j^+(X) - \sum_{j=1}^{n/2} C_j^-(X)\right)" border="0" valign="middle"/>.
</p>

<p align="left">
In other words, classification is based on a majority vote, with the positive clauses voting for <img src="http://latex.codecogs.com/svg.latex?y=1" border="0" valign="middle"/>
 and the negative for <img src="http://latex.codecogs.com/svg.latex?y=0" border="0" valign="middle"/>. The classifier
<p align="center">
<img src="http://latex.codecogs.com/svg.latex?\hat{y} = u\left(x_1 \bar{x}_2 + \bar{x}_1 x_2 - x_1 x_2 - \bar{x}_1 \bar{x}_2\right)" border="0" valign="middle"/>,
</p>
for instance, captures the XOR-relation.
</p>

<p align="left">
A clause is composed by a team of Tsetlin Automata, each Tsetlin Automaton deciding to <i>Exclude</i> or <i>Include</i> a specific literal in the clause. Learning which literals to include is based on reinforcement: Type I feedback is designed to produce frequent patterns, while Type II feedback increases the discrimination power of the patterns.
</p>

<p align="left">
A Tsetlin Machine learns on-line, processing one training example <img src="http://latex.codecogs.com/svg.latex?(X, y)" border="0" valign="middle"/> at a time. <b>Type I feedback</b> is given stochastically to clauses with positive polarity when <img src="http://latex.codecogs.com/svg.latex?y=1" border="0" valign="middle"/> and to clauses with negative polarity when <img src="http://latex.codecogs.com/svg.latex?y=0" border="0" valign="middle"/>. An afflicted clause, <img src="http://latex.codecogs.com/svg.latex?C_j(X)" border="0" valign="middle"/>, in turn, reinforces each of its Tsetlin Automata using two rules:
</p>

* <i>Include</i> is reinforced with probability <img src="http://latex.codecogs.com/svg.latex?\frac{s-1}{s}" border="0" valign="middle"/> (strongly reinforced) if the clause output is <img src="http://latex.codecogs.com/svg.latex?C_j(X)=1" border="0" valign="middle"/>  <b>and</b> the Tsetlin Automaton's literal has value <img src="http://latex.codecogs.com/svg.latex?l_k=1" border="0" valign="middle"/>.
* <i>Exclude</i> is reinforced with probability <img src="http://latex.codecogs.com/svg.latex?\frac{1}{s}" border="0" valign="middle"/> (weakly reinforced) if the clause output is <img src="http://latex.codecogs.com/svg.latex?C_j(X)=0" border="0" valign="middle"/> <b>or</b> the Tsetlin Automaton's literal has value <img src="http://latex.codecogs.com/svg.latex?l_k=0" border="0" valign="middle"/>.

<p>
  <b>Type I feedback</b> is given stochastically to clauses with positive polarity when <img src="http://latex.codecogs.com/svg.latex?y=0" border="0" valign="middle"/> and to clauses with negative polarity when <img src="http://latex.codecogs.com/svg.latex?y=1" border="0" valign="middle"/>.
</p>

<p align="left">
See https://arxiv.org/abs/1804.01508 for details. 
</p>

## Learning Behaviour
The below figure depicts average learning progress (over 50 runs) of the Tsetlin Machine on a binarized, but otherwise unenhanced version of the MNIST dataset (https://en.wikipedia.org/wiki/MNIST_database). See also https://github.com/cair/fast-tsetlin-machine-with-mnist-demo.

<p align="center">
  <img width="75%" src="https://github.com/olegranmo/blob/blob/master/learning_progress.png">
</p>

As seen in the figure, both test and training accuracy increase almost monotonically across the epochs. Even while accuracy on the training data approaches 99.9%, accuracy on the test data continues to increase as well, hitting 98.2% after 400 epochs. This is quite different from what occurs with backpropagation on a neural network, where accuracy on test data starts to drop at some point due to overfitting, without proper regularization mechanisms.

## Noisy XOR Demo

```bash
./NoisyXORDemo.py

Accuracy on test data (no noise): 1.0
Accuracy on training data (40% noise): 0.603

Prediction: x1 = 1, x2 = 0, ... -> y =  1
Prediction: x1 = 0, x2 = 1, ... -> y =  1
Prediction: x1 = 0, x2 = 0, ... -> y =  0
Prediction: x1 = 1, x2 = 1, ... -> y =  0
```

## Requirements

- Python 2.7.x https://www.python.org/downloads/
- Numpy http://www.numpy.org/
- Cython http://cython.org/

## Other Implementations

* High-level Tsetlin Machine Python API with fast C-extensions. Implements the Tsetlin Machine, Convolutional Tsetlin Machine, Regression Tsetlin Machine, Weighted Tsetlin Machine, and Embedding Tsetlin Machine, with support for continuous features, multi-granular clauses, and clause indexing, https://github.com/cair/pyTsetlinMachine, https://pypi.org/project/pyTsetlinMachine/
* Multi-threaded implementation of the Tsetlin Machine, Convolutional Tsetlin Machine, Regression Tsetlin Machine, and Weighted Tsetlin Machine, with support for continuous features and multi-granular clauses, https://github.com/cair/pyTsetlinMachineParallel, https://pypi.org/project/pyTsetlinMachineParallel/
* Fast C++ implementation of the Weighted Tsetlin Machine with MNIST-, IMDb-, and Connect-4 demos, https://github.com/adrianphoulady/weighted-tsetlin-machine-cpp
* Fast bit-operation based implementation in C with MNIST demo, https://github.com/cair/fast-tsetlin-machine-with-mnist-demo
* CUDA implementation with IMDB text classification demo, https://github.com/cair/fast-tsetlin-machine-in-cuda-with-imdb-demo
* Hardware implementations, https://github.com/JieGH/Hardware_TM_Demo
* Kivy implementation, https://github.com/DarshanaAbeyrathna/Tsetlin-Machine-Based-AI-Enabled-Mobile-App-for-Forecasting-the-Number-of-Corona-Patients
* C implementation, https://github.com/cair/TsetlinMachineC
* Parallelized C++ implementation with scikit-learn-compatible Python wrapper, https://github.com/WojciechMigda/Tsetlini, https://github.com/WojciechMigda/PyTsetlini
* Rust implementation, https://github.com/KhaledSharif/TsetlinMachine
* Rust implementation with fast bit-operations, including MNIST demo, https://github.com/jcriddle4/tsetlin_rust_mnist
* C++ implementation, https://github.com/222464/TsetlinMachine
* Node.js implementation, https://github.com/anon767/TsetlinMachine
* C# implementation, https://github.com/cokobware/TsetlinMachineCSharp

## Other Architectures

* The Convolutional Tsetlin Machine, https://github.com/cair/convolutional-tsetlin-machine
* The Regression Tsetlin Machine, https://github.com/cair/regression-tsetlin-machine

## Videos

* Tsetlin Machine - A new paradigm for pervasive AI. DATE SCONA Workshop presentation by Adrian Wheeldon, Microsystems Research Group, Newcastle University. https://youtu.be/TaspuovmSR8
* Tsetlin Machine on Iris Data Set Demo with Handheld MignonAI (http://www.mignon.ai). Presentation by Jie Lei, Microsystems Research Group, Newcastle University. https://youtu.be/BzaPGByX-hg
* Predicting and explaining economic growth using real-time interpretable learning. Presentation by Christian D. Blakely, PwC Switzerland. https://youtu.be/J6K7V7V7ayo
* Early detection of breast cancer from a simple blood test. Presentation by Christian D. Blakely, PwC Switzerland. https://youtu.be/FrHN_aRLRug
* Recent advances in Tsetlin Machines. NORA.ai Webinar presentation by Ole-Christoffer Granmo, CAIR, University of Agder. https://youtu.be/GHelDh3bN00

## Tutorials

Convolutional Tsetlin Machine tutorial, https://github.com/cair/convolutional-tsetlin-machine-tutorial

## Acknowledgements

I thank my colleagues from the Centre for Artificial Intelligence Research (CAIR), Lei Jiao, Xuan Zhang, Geir Thore Berge, Darshana Abeyrathna, Saeed Rahimi Gorji, Sondre Glimsdal, Rupsa Saha, Bimal Bhattarai, Rohan K. Yadev, Bernt Viggo Matheussen, Morten Goodwin, Christian Omlin, Vladimir Zadorozhny (University of Pittsburgh), Jivitesh Sharma, and Ahmed Abouzeid, for their contributions to the development of the Tsetlin machine family of techniques. I would also like to thank our House of CAIR partners, Alex Yakovlev, Rishad Shafik, Adrian Wheeldon, Jie Lei, Tousif Rahman (Newcastle University), Jonny Edwards (Temporal Computing), Marco Wiering (University of Groningen), Christian D. Blakely (PwC Switzerland), Adrian Phoulady, Anders Refsdal Olsen, Halvor Smørvik, and Erik Mathisen for their many contributions.

## Tsetlin Machine Papers

```bash
@article{abeyrathna2020deterministic,
  title="{A Novel Multi-Step Finite-State Automaton for Arbitrarily Deterministic Tsetlin Machine Learning}",
  author={K. Darshana Abeyrathna and Ole-Christoffer Granmo and Rishad Shafik and Alex Yakovlev and Adrian Wheeldon and Jie Lei and Morten Goodwin},
  journal = {arXiv preprint arXiv:2007.02114}, year = {2020}
}
```

```bash
@article{wheeldon2020learning, 
  author={Adrian {Wheeldon} and Rishad {Shafik} and Tousif {Rahman} and Jie {Lei} and Alex {Yakovlev} and Ole-Christoffer {Granmo}}, 
  journal={Philosophical Transactions of the Royal Society A},
  title="{Learning Automata based Energy-efficient AI Hardware Design for IoT}",
  year={2020}
}
```

```bash
@InProceedings{shafik2020explainability,
  title="{Explainability and Dependability Analysis of Learning Automata based AI Hardware}",
  author={Rishad {Shafik} and Adrian {Wheeldon} and Alex {Yakovlev}},
  booktitle={IEEE 26th International Symposium on On-Line Testing and Robust System Design (IOLTS)},
  year={2020},
  organization={IEEE}
}
```

```bash
@article{lavrova2020,
  author = {D. S. {Lavrova} and N. N. {Eliseev}},
  title = "{Network Attacks Detection based on Tsetlin Machine}",
  pages = {17-23},
  journal = {Information Security Problems. Computer Systems.}, year = {2020}
}
```

```bash
@article{abeyrathna2020integer,
  author = {Abeyrathna, Kuruge Darshana and Granmo, Ole-Christoffer and Goodwin, Morten},
  title = "{Extending the Tsetlin Machine With Integer-Weighted Clauses for Increased Interpretability}",
  journal = {arXiv preprint arXiv:2005.05131}, year = {2020}
}
```

```bash
@InProceedings{gorji2020indexing,
  title="{Increasing the Inference and Learning Speed of Tsetlin Machines with Clause Indexing}",
  author={Saeed {Gorji} and Ole Christoffer {Granmo} and Sondre {Glimsdal} and Jonathan {Edwards} and Morten {Goodwin}},
  booktitle={International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems},
  year={2020},
  organization={Springer}
}
```

```bash
@InProceedings{abeyrathna2020integerregression,
  title="{A Regression Tsetlin Machine with Integer Weighted Clauses for Compact Pattern Representation,}",
  author={Abeyrathna, Kuruge Darshana and Granmo, Ole-Christoffer and Goodwin, Morten},
  booktitle={International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems},
  year={2020},
  organization={Springer}
}
```

```bash
@InProceedings{phoulady2020weighted, 
  author={Adrian {Phoulady} and Ole-Christoffer {Granmo} and Saeed Rahimi {Gorji} and Hady Ahmady {Phoulady}}, 
  booktitle={Proceedings of the Ninth International Workshop on Statistical Relational AI (StarAI 2020)}, 
  title="{The Weighted Tsetlin Machine: Compressed Representations with Clause Weighting}",
  year={2020}
}
```

```bash
@InProceedings{wheeldon2020pervasive, 
  author={Adrian {Wheeldon} and Rishad {Shafik} and Alex {Yakovlev} and Jonathan {Edwards} and Ibrahim {Haddadi} and Ole-Christoffer {Granmo}}, 
  booktitle={SCONA Workshop at Design, Automation and Test in Europe (DATE 2020)}, 
  title="{Tsetlin Machine: A New Paradigm for Pervasive AI}",
  year={2020}
}
```

```bash
@article{abeyrathna2019nonlinear, 
  author={K. Darshana {Abeyrathna} and Ole-Christoffer {Granmo} and Xuan {Zhang} and Lei {Jiao} and Morten {Goodwin}}, 
  journal={Philosophical Transactions of the Royal Society A},
  title="{The Regression Tsetlin Machine - A Novel Approach to Interpretable Non-Linear Regression}",
  volume={378}, issue={2164},
  year={2019}
}
```

```bash
@InProceedings{gorji2019multigranular,
  author = {Saeed Rahimi {Gorji} and Ole-Christoffer {Granmo} and Adrian {Phoulady} and Morten {Goodwin}},
  title = "{A Tsetlin Machine with Multigranular Clauses}",
  booktitle="Lecture Notes in Computer Science: Proceedings of the Thirty-ninth International Conference on Innovative Techniques and Applications of Artificial Intelligence (SGAI-2019)", year="2019",
  volume = {11927},
  publisher="Springer International Publishing"
}
```

```bash
@article{berge2019text, 
  author={Geir Thore {Berge} and Ole-Christoffer {Granmo} and Tor Oddbjørn {Tveit} and Morten {Goodwin} and Lei {Jiao} and Bernt Viggo {Matheussen}}, 
  journal={IEEE Access}, 
  title="{Using the Tsetlin Machine to Learn Human-Interpretable Rules for High-Accuracy Text Categorization with Medical Applications}",
  volume={7},
  pages={115134-115146}, 
  year={2019}, 
  doi={10.1109/ACCESS.2019.2935416}, 
  ISSN={2169-3536}
}
```

```bash
@article{granmo2019convtsetlin,
  author = {{Granmo}, Ole-Christoffer and {Glimsdal}, Sondre and {Jiao}, Lei and {Goodwin}, Morten and {Omlin}, Christian W. and {Berge}, Geir Thore},
  title = "{The Convolutional Tsetlin Machine}",
  journal = {arXiv preprint arXiv:1905.09688}, year = {2019}
}
```

```bash
@InProceedings{abeyrathna2019regressiontsetlin,
  author = {{Abeyrathna}, Kuruge Darshana and {Granmo}, Ole-Christoffer and {Jiao}, Lei and {Goodwin}, Morten},
  title = "{The Regression Tsetlin Machine: A Tsetlin Machine for Continuous Output Problems}",
  editor="Moura Oliveira, Paulo and Novais, Paulo and Reis, Lu{\'i}s Paulo ",
  booktitle="Progress in Artificial Intelligence", year="2019",
  publisher="Springer International Publishing",
  pages="268--280"
}
```

```bash
@InProceedings{abeyrathna2019continuousinput,
  author = {{Abeyrathna}, Kuruge Darshana and {Granmo}, Ole-Christoffer and {Zhang}, Xuan and {Goodwin}, Morten},
  title = "{A Scheme for Continuous Input to the Tsetlin Machine with Applications to Forecasting Disease Outbreaks}",
  booktitle = "{Advances and Trends in Artificial Intelligence. From Theory to Practice}", year = "2019",
  editor = "Wotawa, Franz and Friedrich, Gerhard and Pill, Ingo and Koitz-Hristov, Roxane and Ali, Moonis",
  publisher = "Springer International Publishing",
  pages = "564--578"
}
```

```bash
@article{granmo2018tsetlin,
  author = {{Granmo}, Ole-Christoffer},
  title = "{The Tsetlin Machine - A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic}",
  journal = {arXiv preprint arXiv:1804.01508}, year = {2018}
}
```

## Licence

Copyright (c) 2020 Ole-Christoffer Granmo

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
