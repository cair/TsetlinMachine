# Tsetlin Machine
![License](https://img.shields.io/github/license/microsoft/interpret.svg?style=flat-square) ![Maintenance](https://img.shields.io/maintenance/yes/2024?style=flat-square)

Code and datasets for the Tsetlin Machine. Implements the Tsetlin Machine from https://arxiv.org/abs/1804.01508, including the multiclass version. The Tsetlin Machine solves complex pattern recognition problems with easy-to-interpret propositional formulas, composed by a collective of Tsetlin Automata.

<p align="center">
  <img width="75%" src="https://github.com/olegranmo/blob/blob/master/Tsetlin_Machine_Example_Configuration_Full.png">
</p>

## Contents

- [Basics](#basics)
  - [Classification](#classification)
  - [Learning](#learning)
  - [Learning Behaviour](#learning-behaviour)
- [Noisy XOR Demo](#noisy-xor-demo)
- [Requirements](#requirements)
- [Other Implementations](#other-implementations)
- [Other Architectures](#other-architectures)
- [Hardware](#hardware)
- [Books](#books)
- [Conferences](#conferences)
- [Videos](#videos)
- [Tutorials](#tutorials)
- [Acknowledgements](#acknowledgements)
- [Tsetlin Machine Papers](#tsetlin-machine-papers)
- [Licence](#licence)

## Basics

### Classification

<p align="left">
A basic Tsetlin Machine takes a vector <img src="http://latex.codecogs.com/svg.latex?X%3D%5Bx_1%2C%5Cldots%2Cx_o%5D" border="0" valign="middle"/> of Boolean features as input, to be classified into one of two classes, <img src="http://latex.codecogs.com/svg.latex?y=0" border="0" valign="middle"/> or <img src="http://latex.codecogs.com/svg.latex?y=1" border="0" valign="middle"/>. Together with their negated counterparts, <img src="http://latex.codecogs.com/svg.latex?\bar{x}_k = {\lnot} {x}_k = 1-x_k" border="0" valign="middle"/>, the features form a literal set <img src="http://latex.codecogs.com/svg.latex?L = \{x_1,\ldots,x_o,\bar{x}_1,\ldots,\bar{x}_o\}" border="0" valign="middle"/>.
</p>

<p align="left">
A Tsetlin Machine pattern is formulated as a conjunctive clause <img src="http://latex.codecogs.com/svg.latex?C_j" border="0" valign="middle"/>, formed by ANDing a subset <img src="http://latex.codecogs.com/svg.latex?L_j {\subseteq} L" border="0" valign="middle"/> of the literal set:
</p>

<p align="center">
  <img src="http://latex.codecogs.com/svg.latex?C_j (X)=\bigwedge_{{{l}} {\in} L_j} l = \prod_{{{l}} {\in} L_j} l" border="0" valign="middle"/>.
</p>

<p align="left">
For example, the clause <img src="http://latex.codecogs.com/svg.latex?C_j(X)=x_1\land{\lnot}x_2=x_1 \bar{x}_2" border="0" valign="middle"/> consists of the literals <img src="http://latex.codecogs.com/svg.latex?L_j = \{x_1,\bar{x}_2\}" border="0" valign="middle"/> and outputs <img src="http://latex.codecogs.com/svg.latex?1" border="0" valign="middle"/> iff <img src="http://latex.codecogs.com/svg.latex?x_1 = 1" border="0" valign="middle"/> and <img src="http://latex.codecogs.com/svg.latex?x_2 = 0" border="0" valign="middle"/>.
</p>

<p align="left">
The number of clauses employed is a user-configurable parameter <img src="http://latex.codecogs.com/svg.latex?n" border="0"/>. Half of the clauses are assigned positive polarity. The other half is assigned negative polarity. The clause outputs, in turn, are combined into a classification decision through summation and thresholding using the unit step function <img src="http://latex.codecogs.com/svg.latex?{u(v) = 1 ~\mathbf{if}~ v \ge 0 ~\mathbf{else}~ 0}" border="0" valign="middle"/>:
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

### Learning

<p align="left">
A clause <img src="http://latex.codecogs.com/svg.latex?C_j(X)" border="0" valign="middle"/> is composed by a team of Tsetlin Automata, each Tsetlin Automaton deciding to <i>Include</i> or <i>Exclude</i> a specific literal <img src="http://latex.codecogs.com/svg.latex?l_k" border="0" valign="middle"/> in the clause (see figure above). Learning which literals to include is based on reinforcement: Type I feedback produces frequent patterns, while Type II feedback increases the discrimination power of the patterns.
</p>

<p align="left">
A Tsetlin Machine learns on-line, processing one training example <img src="http://latex.codecogs.com/svg.latex?{(X, y)}" border="0" valign="middle"/> at a time.
</p>

<p align="center">
  <img width="52%" src="https://github.com/olegranmo/blob/blob/master/Type_I_Feedback.png">
</p>

<p align="left">
<b>Type I feedback</b> is given stochastically to clauses with positive polarity when <img src="http://latex.codecogs.com/svg.latex?y=1" border="0" valign="middle"/> and to clauses with negative polarity when <img src="http://latex.codecogs.com/svg.latex?y=0" border="0" valign="middle"/>. An afflicted clause, in turn, reinforces each of its Tsetlin Automata based on: (i) the clause output <img src="http://latex.codecogs.com/svg.latex?C_j(X)" border="0" valign="middle"/>; (ii) the action of the targeted Tsetlin Automaton - <i>Include</i> or <i>Exclude</i>; and (iii) the value of the literal <img src="http://latex.codecogs.com/svg.latex?l_k" border="0" valign="middle"/> assigned to the automaton. As shown in Table 1, two rules govern Type I feedback, given independently to each Tsetlin Automaton of the clause:
</p>

* <i>Include</i> is rewarded and <i>Exclude</i> is penalized with probability <img src="http://latex.codecogs.com/svg.latex?\frac{s-1}{s}" border="0" valign="middle"/> whenever <img src="http://latex.codecogs.com/svg.latex?C_j(X)=1" border="0" valign="middle"/> <b>and</b> <img src="http://latex.codecogs.com/svg.latex?{l_k=1}" border="0" valign="middle"/>. This reinforcement is strong (triggers with high probability) and makes the clause remember and refine the pattern it recognizes in <img src="http://latex.codecogs.com/svg.latex?X" border="0" valign="middle"/>. 
* <i>Include</i> is penalized and <i>Exclude</i> is rewarded with probability <img src="http://latex.codecogs.com/svg.latex?\frac{1}{s}" border="0" valign="middle"/> if <img src="http://latex.codecogs.com/svg.latex?C_j(X)=0" border="0" valign="middle"/> <b>or</b> <img src="http://latex.codecogs.com/svg.latex?{l_k}=0" border="0" valign="middle"/>. This reinforcement is weak (triggers with low probability) and coarsens infrequent patterns, making them frequent.

<p>
Above, <img src="http://latex.codecogs.com/svg.latex?s" border="0" valign="middle"/> is a hyperparameter that controls the frequency of the patterns produced.
</p>

<p align="center">
  <img width="52%" src="https://github.com/olegranmo/blob/blob/master/Type_II_Feedback.png">
</p>

<p>
<b>Type II feedback</b> is given stochastically to clauses with positive polarity when <img src="http://latex.codecogs.com/svg.latex?y=0" border="0" valign="middle"/> and to clauses with negative polarity when <img src="http://latex.codecogs.com/svg.latex?y=1" border="0" valign="middle"/>. Again, an affected clause reinforces each of its Tsetlin Automata based on: (i) the clause output <img src="http://latex.codecogs.com/svg.latex?C_j(X)" border="0" valign="middle"/>; (ii) the action of the targeted Tsetlin Automaton - <i>Include</i> or <i>Exclude</i>; and (iii) the value of the literal <img src="http://latex.codecogs.com/svg.latex?l_k" border="0" valign="middle"/> assigned to the automaton. As captured by Table 2, Type II feedback penalizes <i>Exclude</i> whenever <img src="http://latex.codecogs.com/svg.latex?C_j(X)=1" border="0" valign="middle"/> <b>and</b> <img src="http://latex.codecogs.com/svg.latex?{l_k=0}" border="0" valign="middle"/>. This feedback is strong and produces candidate literals for discriminating between <img src="http://latex.codecogs.com/svg.latex?y=0" border="0" valign="middle"/> and <img src="http://latex.codecogs.com/svg.latex?{y=1}" border="0" valign="middle"/>.
</p>

<p>
<b>Resource allocation</b> dynamics ensure that clauses distribute themselves across the frequent patterns, rather than missing some and overconcentrating on others. That is, for any input <img src="http://latex.codecogs.com/svg.latex?X" border="0" valign="middle"/>, the probability of reinforcing a clause gradually drops to zero as the clause output sum
</p>
<p align="center">
  <img src="http://latex.codecogs.com/svg.latex?v = \sum_{j=1}^{n/2} C_j^+(X) - \sum_{j=1}^{n/2} C_j^-(X)" border="0" valign="middle"/>
</p>
<p>
approaches a user-set target <img src="http://latex.codecogs.com/svg.latex?T" border="0" valign="middle"/> for <img src="http://latex.codecogs.com/svg.latex?y=1" border="0" valign="middle"/> (<img src="http://latex.codecogs.com/svg.latex?-T" border="0" valign="middle"/> for <img src="http://latex.codecogs.com/svg.latex?y=0" border="0" valign="middle"/>). To exemplify, the below plot shows the probability of reinforcing a clause when <img src="http://latex.codecogs.com/svg.latex?y=1" border="0" valign="middle"/> and <img src="http://latex.codecogs.com/svg.latex?T=2" border="0" valign="middle"/> for different clause output sums <img src="http://latex.codecogs.com/svg.latex?v" border="0" valign="middle"/>:
<p/>
<p align="center">
  <img width="60%" src="https://github.com/olegranmo/blob/blob/master/Clause_Activation_Probability_y1.png">
</p>
<p>
If a clause is not reinforced, it does not give feedback to its Tsetlin Automata, and these are thus left unchanged.  In the extreme, when the voting sum <img src="http://latex.codecogs.com/svg.latex?v" border="0" valign="middle"/> equals or exceeds the target <img src="http://latex.codecogs.com/svg.latex?T" border="0" valign="middle"/> (the Tsetlin Machine has successfully recognized the input <img src="http://latex.codecogs.com/svg.latex?X" border="0" valign="middle"/>), no clauses are reinforced. Accordingly, they are free to learn new patterns, naturally balancing the pattern representation resources.
</p>

<p>
See https://arxiv.org/abs/1804.01508 for details. 
</p>

### Learning Behaviour
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

* Tsetlin Machine Unified - One Codebase to Rule Them All. Implements the Tsetlin Machine, Coalesced Tsetlin Machine, Convolutional Tsetlin Machine, Regression Tsetlin Machine, and Weighted Tsetlin Machine, with support for continuous features, drop clause, Type III Feedback, focused negative sampling, multi-task classifier, autoencoder, literal budget, and one-vs-one multi-class classifier. TMU is written in Python with wrappers for C and CUDA-based clause evaluation and updating, https://github.com/cair/tmu, https://pypi.org/project/tmu/
* Massively Parallel and Asynchronous Architecture for Logic-based AI. Implements the Tsetlin Machine, Convolutional Tsetlin Machine, Regression Tsetlin Machine, and Weighted Tsetlin Machine, with support for continuous features, https://github.com/cair/PyTsetlinMachineCUDA, https://pypi.org/project/PyTsetlinMachineCUDA/
* High-level Tsetlin Machine Python API with fast C-extensions. Implements the Tsetlin Machine, Convolutional Tsetlin Machine, Regression Tsetlin Machine, Weighted Tsetlin Machine, and Embedding Tsetlin Machine, with support for continuous features, multi-granular clauses, and clause indexing, https://github.com/cair/pyTsetlinMachine, https://pypi.org/project/pyTsetlinMachine/
* Multi-threaded implementation of the Tsetlin Machine, Convolutional Tsetlin Machine, Regression Tsetlin Machine, and Weighted Tsetlin Machine, with support for continuous features and multi-granular clauses, https://github.com/cair/pyTsetlinMachineParallel, https://pypi.org/project/pyTsetlinMachineParallel/
* High-performance Tsetlin Machine implementation in Julia with batching and low-level parallelization (52 million classifications per second on MNIST), https://github.com/BooBSD/Tsetlin.jl
* Class-parallel Tsetlin Machine in C with logging support, https://github.com/ashurrafiev/ClassParallelTM
* GUI application for visualising Tsetlin machine states and learning process, https://github.com/ashurrafiev/TsetlinMachineViewer
* Fast C++ implementation of the Weighted Tsetlin Machine with MNIST-, IMDb-, and Connect-4 demos, https://github.com/adrianphoulady/weighted-tsetlin-machine-cpp
* Fast bit-operation based implementation in C with MNIST demo, https://github.com/cair/fast-tsetlin-machine-with-mnist-demo
* CUDA implementation with IMDB text classification demo, https://github.com/cair/fast-tsetlin-machine-in-cuda-with-imdb-demo
* Hardware implementations, https://github.com/JieGH/Hardware_TM_Demo
* Kivy implementation, https://github.com/DarshanaAbeyrathna/Tsetlin-Machine-Based-AI-Enabled-Mobile-App-for-Forecasting-the-Number-of-Corona-Patients
* C implementation, https://github.com/cair/TsetlinMachineC
* Parallelized C++ implementation of Multi-class and Regression Tsetlin Machine with scikit-learn-compatible Python wrapper, https://github.com/WojciechMigda/Tsetlini, https://github.com/WojciechMigda/PyTsetlini
* Rust implementation, https://github.com/KhaledSharif/TsetlinMachine
* Rust implementation with fast bit-operations, including MNIST demo, https://github.com/jcriddle4/tsetlin_rust_mnist
* C++ implementation, https://github.com/222464/TsetlinMachine
* Node.js implementation, https://github.com/anon767/TsetlinMachine
* C# implementation, https://github.com/cokobware/TsetlinMachineCSharp
* F# implementation, https://github.com/fwaris/FsTsetlin

## Other Architectures

* The Convolutional Tsetlin Machine, https://github.com/cair/convolutional-tsetlin-machine
* The Regression Tsetlin Machine, https://github.com/cair/regression-tsetlin-machine
* Coalesced Multi-Output Tsetlin Machine, https://github.com/cair/PyCoalescedTsetlinMachineCUDA/
* Massively Parallel and Asynchronous Architecture for Logic-based AI, https://github.com/cair/PyTsetlinMachineCUDA

## Hardware

* [Literal Labs](https://www.literal-labs.ai)

## Books

* [An Introduction to Tsetlin Machines](https://tsetlinmachine.org)

## Conferences

* [International Symposium on the Tsetlin Machine (ISTM)](https://istm.no) [Proceedings [2022](https://ieeexplore.ieee.org/xpl/conhome/9923753/proceeding), [2023](https://ieeexplore.ieee.org/xpl/conhome/10454903/proceeding)]

## Videos

* Watching the state transitions of Tsetlin Automata. Demo by Jie Lei, Microsystems Research Group, Newcastle University. https://youtu.be/wXyiLtlpwHI
* Keyword Spotting using Tsetlin Machines. Demo by Jie Lei, Microsystems Research Group, Newcastle University. https://youtu.be/JW0tztpjX8k
* Mignon AI Presentation at Arm Summit 2020 by Adrian Wheeldon and Jie Lei, Microsystems Research Group, Newcastle University. https://youtu.be/N-wkgibJAZE
* Explainability and Dependability Analysis of Learning Automata based AI Hardware. IOLTS presentation by Rishad Shafik, Microsystems Research Group, Newcastle University. https://youtu.be/IjzZY0fDYiA
* Tsetlin Machine - A new paradigm for pervasive AI. DATE SCONA Workshop presentation by Adrian Wheeldon, Microsystems Research Group, Newcastle University. https://youtu.be/TaspuovmSR8
* Quick Guide to the Tsetlin Machine using Sequential Logic in Logisim. Presentations by Jie Lei, Microsystems Research Group, Newcastle University.
  * Tsetlin Automaton - https://youtu.be/XzWSPo7GF94
  * Clause Calculation - https://youtu.be/Yfrt-W40LiI
  * Summation and Thresholding - https://youtu.be/ipKHuHMDafU
* Tsetlin Machine on Iris Data Set Demo with Handheld MignonAI (http://www.mignon.ai). Presentation by Jie Lei, Microsystems Research Group, Newcastle University. https://youtu.be/BzaPGByX-hg
* The-Ruler-of-Tsetlin-Automaton. Presentation by Jie Lei, Microsystems Research Group, Newcastle University. https://youtu.be/LltDhg4ZuWo
* Interpretable clustering and dimension reduction with Tsetlin automata machine learning. Presentation by Christian D. Blakely, PwC Switzerland. https://youtu.be/5-09LOGLcV8
* Predicting and explaining economic growth using real-time interpretable learning. Presentation by Christian D. Blakely, PwC Switzerland. https://youtu.be/J6K7V7V7ayo
* Early detection of breast cancer from a simple blood test. Presentation by Christian D. Blakely, PwC Switzerland. https://youtu.be/FrHN_aRLRug
* Recent advances in Tsetlin Machines. NORA.ai Webinar presentation by Ole-Christoffer Granmo, CAIR, University of Agder. https://youtu.be/GHelDh3bN00

## Tutorials

Convolutional Tsetlin Machine tutorial, https://github.com/cair/convolutional-tsetlin-machine-tutorial

## Acknowledgements

I thank my colleagues from the Centre for Artificial Intelligence Research (CAIR), Lei Jiao, Xuan Zhang, Geir Thore Berge, Darshana Abeyrathna, Saeed Rahimi Gorji, Sondre Glimsdal, Rupsa Saha, Bimal Bhattarai, Rohan K. Yadav, Bernt Viggo Matheussen, Morten Goodwin, Christian Omlin, Vladimir Zadorozhny (University of Pittsburgh), Jivitesh Sharma, Ahmed Abouzeid, and Charul Giri, for their contributions to the development of the Tsetlin machine family of techniques. I would also like to thank our House of CAIR partners, Alex Yakovlev, Rishad Shafik, Ashur Rafiev, Sidharth Maheshwari, Adrian Wheeldon, Jie Lei, Tousif Rahman,  (Newcastle University), Jonny Edwards (Temporal Computing), Marco Wiering (University of Groningen), Christian D. Blakely (PwC Switzerland), Adrian Phoulady, Anders Refsdal Olsen, Halvor Smørvik, and Erik Mathisen for their many contributions.

## Tsetlin Machine Papers

```bash
@InProceedings{yadav2022robust,
  title     = {Robust Interpretable Text Classification against Spurious Correlations Using AND-rules with Negation},
  author    = {Yadav, Rohan Kumar and Jiao, Lei and Granmo, Ole-Christoffer and Goodwin, Morten},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Lud De Raedt},
  pages     = {4439--4446},
  year      = {2022},
  month     = {7},
  note      = {Main Track}
  doi       = {10.24963/ijcai.2022/616},
  url       = {https://doi.org/10.24963/ijcai.2022/616},
}
```

```bash
@InProceedings{bhattarai2022fakenews,
  author    = {Bhattarai, Bimal  and  Granmo, Ole-Christoffer  and  Jiao, Lei},
  title     = {Explainable Tsetlin Machine Framework for Fake News Detection with Credibility Score Assessment},
  booktitle      = {Proceedings of the Language Resources and Evaluation Conference},
  month          = {June},
  year           = {2022},
  address        = {Marseille, France},
  publisher      = {European Language Resources Association},
  pages     = {4894--4903},
  url       = {https://aclanthology.org/2022.lrec-1.523}
}
```

```bash
@InProceedings{bhattarai2022convtext,
  author    = {Bhattarai, Bimal  and  Granmo, Ole-Christoffer  and  Jiao, Lei},
  title     = {ConvTextTM: An Explainable Convolutional Tsetlin Machine Framework for Text Classification},
  booktitle      = {Proceedings of the Language Resources and Evaluation Conference},
  month          = {June},
  year           = {2022},
  address        = {Marseille, France},
  publisher      = {European Language Resources Association},
  pages     = {3761--3770},
  url       = {https://aclanthology.org/2022.lrec-1.401}
}
```


```bash
@article{saha2021disc,
  author = {Saha, Rupsa and Granmo, Ole-Christoffer and Goodwin, Morten},
  title = {Using Tsetlin Machine to discover interpretable rules in natural language processing applications},
  journal = {Expert Systems},
  url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/exsy.12873},
  year={2021}
}
```

```bash
@article{abeyrathna2021multistep,
  author = {Abeyrathna, Kuruge Darshana and Granmo, Ole-Christoffer and Shafik, Rishad and Jiao, Lei and Wheeldon, Adrian and Yakovlev, Alex and Lei, Jie and Goodwin, Morten},
  title = {A multi-step finite-state automaton for arbitrarily deterministic Tsetlin Machine learning},
  journal = {Expert Systems},
  url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/exsy.12836},
  year={2021}
}
```

```bash
@InProceedings{yadav2021dwr,
  title={Enhancing Interpretable Clauses Semantically using Pretrained Word Representation},
  author={Rohan Kumar Yadav and Lei Jiao and Ole-Christoffer Granmo and Morten Goodwin},
  booktitle={BLACKBOXNLP},
  url = {https://aclanthology.org/2021.blackboxnlp-1.19.pdf},
  year={2021}
}
```

```bash
@article{jiao2021andorconvergence,
  title={On the Convergence of Tsetlin Machines for the AND and the OR Operators},
  author={Lei Jiao and Xuan Zhang and Ole-Christoffer Granmo},
  journal = {arXiv preprint arXiv:2109.09488}, year = {2021},
  url = {https://arxiv.org/abs/2109.09488}
}
```

```bash
@InProceedings{wheeldon2021self,
  title="{Self-Timed Reinforcement Learning using Tsetlin Machine}",
  author={Adrian Wheeldon and Alex Yakovlev and Rishad Shafik},
  booktitle={27th IEEE International Symposium on Asynchronous Circuits and Systems (ASYNC 2021)},
  year={2021},
  organization={IEEE},
  url={https://arxiv.org/abs/2109.00846}
}
```

```bash
@article{glimsdal2021coalesced,
  title={Coalesced Multi-Output Tsetlin Machines with Clause Sharing},
  author={Sondre Glimsdal and Ole-Christoffer Granmo},
  journal = {arXiv preprint arXiv:2108.07594}, year = {2021},
  url = {https://arxiv.org/abs/2108.07594}
}
```

```bash
@article{Abeyrathna2021adaptivesparse,
  title="{Adaptive Sparse Representation of Continuous Input for Tsetlin Machines Based on Stochastic Searching on the Line}",
  volume={10},
  ISSN={2079-9292},
  url={http://dx.doi.org/10.3390/electronics10172107},
  DOI={10.3390/electronics10172107},
  number={17},
  journal={Electronics},
  publisher={MDPI AG},
  author={Abeyrathna, Kuruge Darshana and Granmo, Ole-Christoffer and Goodwin, Morten},
  year={2021},
  month={Aug},
  pages={2107}}
```

```bash
@article{zhang2021convergence,
  title = {On the {{Convergence}} of {{Tsetlin Machines}} for the {{IDENTITY}}- and {{NOT Operators}}},
  author = {Zhang, Xuan and Jiao, Lei and Granmo, Ole-Christoffer and Goodwin, Morten},
  year = {2021},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence}
}
```

```bash
@InProceedings{abeyrathna2021parallel,
  title="{Massively Parallel and Asynchronous Tsetlin Machine Architecture Supporting Almost Constant-Time Scaling}",
  author={K. Darshana Abeyrathna and Bimal Bhattarai and Morten Goodwin and Saeed Gorji and Ole-Christoffer Granmo and Lei Jiao and Rupsa Saha and Rohan K. Yadav},
  booktitle={The Thirty-eighth International Conference on Machine Learning (ICML 2021)},
  year={2021},
  organization={ICML}
}
```

```bash
@article{sharma2021dropclause,
  title="{Human Interpretable AI: Enhancing Tsetlin Machine Stochasticity with Drop Clause}",
  author={Jivitesh Sharma and Rohan Yadav and Ole-Christoffer Granmo and Lei Jiao},
  journal = {arXiv preprint arXiv:2105.14506}, year = {2021},
  url = {https://arxiv.org/abs/2105.14506}
}
```

```bash
@article{bhattarai2021fakenews,
  title="{Explainable Tsetlin Machine framework for fake news detection with credibility score assessment}",
  author={Bimal Bhattarai and Ole-Christoffer Granmo and Lei Jiao},
  journal = {arXiv preprint arXiv:2105.09114}, year = {2021},
  url = {https://arxiv.org/abs/2105.09114}
}
```

```bash
@article{bhattarai2021wordlevel,
  title="{Word-level Human Interpretable Scoring Mechanism for Novel Text Detection Using Tsetlin Machines}",
  author={Bimal Bhattarai and Ole-Christoffer Granmo and Lei Jiao},
  journal = {arXiv preprint arXiv:2105.04708}, year = {2021},
  url = {https://arxiv.org/abs/2105.04708}
}
```

```bash
@article{lei2021kws,
  title="{Low-Power Audio Keyword Spotting Using Tsetlin Machines}",
  author={Jie Lei and Tousif Rahman and Rishad Shafik and Adrian Wheeldon and Alex Yakovlev and Ole-Christoffer Granmo and Fahim Kawsar and Akhil Mathur},
  journal = {Journal of Low Power Electronics and Applications}, year = {2021},
  volume=11,
  issue=18,
  url = {https://www.mdpi.com/2079-9268/11/2/18},
  organization={MDPI}
}
```

```bash
@InProceedings{blakely2021closed,
  title="{Closed-Form Expressions for Global and Local Interpretation of Tsetlin Machines}",
  author={Christian D. {Blakely} and Ole-Christoffer {Granmo}},
  booktitle={34th International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems (IEA/AIE 2021)},
  year={2021},
  organization={Springer}
}
```

```bash
@InProceedings{gorji2021rl,
  title="{Explainable Reinforcement Learning with the Tsetlin Machine}",
  author={Saeed {Gorji} and Ole Christoffer {Granmo} and Marco {Wiering}},
  booktitle={34th International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems (IEA/AIE 2021)},
  year={2021},
  organization={Springer}
}
```

```bash
@InProceedings{yadav2021sentiment,
  title="{Human-Level Interpretable Learning for Aspect-Based Sentiment Analysis}",
  author={Rohan Kumar {Yadav} and Lei {Jiao} and Ole-Christoffer {Granmo} and Morten {Goodwin}},
  booktitle={The Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI-21)},
  year={2021},
  organization={AAAI}
}
```

```bash
@InProceedings{nicolae2021question,
  title="{Question Classification using Interpretable Tsetlin Machine}",
  author={Dragos Constantin {Nicolae}},
  booktitle={The 1st International Workshop on Machine Reasoning (MRC 2021)},
  year={2021}
}
```

```bash
@article{saha2021relational,
  title="{A Relational Tsetlin Machine with Applications to Natural Language Understanding}",
  author={Rupsa Saha and Ole-Christoffer Granmo and Vladimir I. Zadorozhny and Morten Goodwin},
  journal = {arXiv preprint arXiv:2102.10952}, year = {2021},
  url = {https://arxiv.org/abs/2102.10952}
}
```

```bash
@InProceedings{yadav2021wordsense,
  title="{Interpretability in Word Sense Disambiguation using Tsetlin Machine}",
  author={Rohan Kumar {Yadav} and Lei {Jiao} and Ole-Christoffer {Granmo} and Morten {Goodwin}},
  booktitle={13th International Conference on Agents and Artificial Intelligence (ICAART 2021)},
  year={2021},
  organization={INSTICC}
}
```

```bash
@InProceedings{bhattarai2021novelty,
  title="{Measuring the Novelty of Natural Language Text Using the Conjunctive Clauses of a Tsetlin Machine Text Classifier}",
  author={Bimal Bhattarai and Lei Jiao and Ole-Christoffer Granmo},
  booktitle={13th International Conference on Agents and Artificial Intelligence (ICAART 2021)},
  year={2021},
  organization={INSTICC}
}
```

```bash
@InProceedings{abeyrathna2021convreg,
  title="{Convolutional Regression Tsetlin Machine}",
  author={Abeyrathna, Kuruge Darshana and Granmo, Ole-Christoffer and Goodwin, Morten},
  booktitle={6th International Conference on Machine Learning Technologies (ICMLT 2021)},
  year={2021},
  organization={ACM}
}
```

```bash
@article{abeyrathna2021integer,
  author = {Abeyrathna, Kuruge Darshana and Granmo, Ole-Christoffer and Goodwin, Morten},
  title = "{Extending the Tsetlin Machine With Integer-Weighted Clauses for Increased Interpretability}",
  journal = {IEEE Access},
  volume = 9,
  year = {2021}
}
```

```bash
@article{jiao2021xor,
  title="{On the Convergence of Tsetlin Machines for the XOR Operator}",
  author={Lei Jiao and Xuan Zhang and Ole-Christoffer Granmo and K. Darshana Abeyrathna},
  journal = {arXiv preprint arXiv:2101.02547}, year = {2021},
  url = {https://arxiv.org/abs/2101.02547}
}
```

```bash
@InProceedings{wheeldon2021low,
  title="{Low-Latency Asynchronous Logic Design for Inference at the Edge}",
  author={Adrian Wheeldon and Alex Yakovlev and Rishad Shafik and Jordan Morris},
  booktitle={2021 Design, Automation and Test in Europe Conference (DATE21)},
  year={2021},
  pages="370-373"
}
```

```bash
@InProceedings{lei2020arithmetic,
  title="{From Arithmetic to Logic Based AI: A Comparative Analysis of Neural Networks and Tsetlin Machine}",
  author={Jie {Lei} and Adrian {Wheeldon} and Rishad {Shafik} and Alex {Yakovlev} and Ole-Christoffer {Granmo}},
  booktitle={27th IEEE International Conference on Electronics Circuits and Systems (ICECS2020)},
  year={2020},
  organization={IEEE}
}
```

```bash
@InProceedings{abeyrathna2020auc,
  title="{On Obtaining Classification Confidence, Ranked Predictions and AUC with Tsetlin Machines}",
  author={K. Darshana Abeyrathna and Ole-Christoffer Granmo and Morten Goodwin},
  booktitle={2020 IEEE Symposium Series on Computational Intelligence (SSCI)},
  year={2020},
  organization={IEEE}
}
```

```bash
@InProceedings{abeyrathna2020intrusion,
  title="{Intrusion Detection with Interpretable Rules Generated Using the Tsetlin Machine}",
  author={K. Darshana Abeyrathna and Harsha S. Gardiyawasam Pussewalage and Sasanka N. Ranasinghea and Vladimir A. Oleshchuk and Ole-Christoffer Granmo},
  booktitle={2020 IEEE Symposium Series on Computational Intelligence (SSCI)},
  year={2020},
  organization={IEEE}
}
```

```bash
@InProceedings{abeyrathna2020adaptive,
  title="{Adaptive Continuous Feature Binarization for Tsetlin Machines Applied to Forecasting Dengue Incidences in the Philippines}",
  author={K. Darshana Abeyrathna and Ole-Christoffer Granmo and Xuan Zhang and Morten Goodwin},
  booktitle={2020 IEEE Symposium Series on Computational Intelligence (SSCI)},
  year={2020},
  organization={IEEE}
}
```

```bash
@InProceedings{saha2020causal,
  author = {Rupsa {Saha} and Ole-Christoffer {Granmo} and Morten {Goodwin}},
  title = "{Mining Interpretable Rules for Sentiment and Semantic Relation Analysis using Tsetlin Machines}",
  booktitle="Lecture Notes in Computer Science: Proceedings of the 40th International Conference on Innovative Techniques and Applications of Artificial Intelligence (SGAI-2020)", year="2020",
  publisher="Springer International Publishing"
}
```

```bash
@InProceedings{abeyrathna2020deterministic,
  title="{A Novel Multi-Step Finite-State Automaton for Arbitrarily Deterministic Tsetlin Machine Learning}",
  author={K. Darshana Abeyrathna and Ole-Christoffer Granmo and Rishad Shafik and Alex Yakovlev and Adrian Wheeldon and Jie Lei and Morten Goodwin},
  booktitle="Lecture Notes in Computer Science: Proceedings of the 40th International Conference on Innovative Techniques and Applications of Artificial Intelligence (SGAI-2020)", year="2020",
  publisher="Springer International Publishing"
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
@InProceedings{gorji2020indexing,
  title="{Increasing the Inference and Learning Speed of Tsetlin Machines with Clause Indexing}",
  author={Saeed {Gorji} and Ole Christoffer {Granmo} and Sondre {Glimsdal} and Jonathan {Edwards} and Morten {Goodwin}},
  booktitle={33rd International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems (IEA/AIE 2020)},
  year={2020},
  organization={Springer}
}
```

```bash
@InProceedings{abeyrathna2020integerregression,
  title="{A Regression Tsetlin Machine with Integer Weighted Clauses for Compact Pattern Representation}",
  author={Abeyrathna, Kuruge Darshana and Granmo, Ole-Christoffer and Goodwin, Morten},
  booktitle={33rd International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems (IEA/AIE 2020)},
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
@article{abeyrathna2020nonlinear, 
  author={K. Darshana {Abeyrathna} and Ole-Christoffer {Granmo} and Xuan {Zhang} and Lei {Jiao} and Morten {Goodwin}}, 
  journal={Philosophical Transactions of the Royal Society A},
  title="{The Regression Tsetlin Machine - A Novel Approach to Interpretable Non-Linear Regression}",
  volume={378}, issue={2164},
  year={2020}
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
  journal = {arXiv preprint arXiv:1905.09688}, year = {2019},
  url={https://arxiv.org/abs/1905.09688}
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
  journal = {arXiv preprint arXiv:1804.01508}, year = {2018},
  url={https://arxiv.org/abs/1804.01508}
}
```

## Licence

Copyright (c) 2023 Ole-Christoffer Granmo

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
