# [Fast.ai](https://www.usfca.edu/data-institute/certificates/deep-learning-part-two) Deep Learning from the Foundations (Spring 2019)
*Part II of Fast.ai's two-part deep learning course, offered through [The Data Institute at USF](https://www.usfca.edu/data-institute). From March through the end of April in 2019. Part I is [here](https://github.com/jamesdellinger/fastai_deep_learning_course_part1_v3).*

A bottom-up approach (through code, not math equations) to becoming an expert deep learning practitioner and experimenter. 

We implemented core [fastai](https://github.com/fastai/fastai) and [PyTorch](https://pytorch.org/docs/stable/index.html) classes and modules from scratch, achieving similar or better performance. We also practiced coding up techniques introduced in various papers, and then spent significant time on strategies useful in decreasing model training time (parallelization, JIT).

The final two weeks were spent diving deep into [Swift for TensorFlow](https://www.tensorflow.org/swift) with [Chris Lattner](http://www.nondot.org/sabre/), where we saw first-hand how [differentiable programming](https://medium.com/@karpathy/software-2-0-a64152b37c35) could work, and experienced the joy of coding deep learning models in a language that actually gets sent directly to the compiler.

All in all, I came away with both the know-how to engineer cutting-edge deep learning ideas from scratch with optimized code, as well as the expertise necessary to research and explore new ideas of my own.

## My Reimplementations of Lesson Notebooks
### Week 8: Building Optimized Matmul, Forward and Backpropagation from Scratch
* [Matrix Multiplication](http://nbviewer.jupyter.org/github/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/01_matmul_my_reimplementation.ipynb?flush_cache=true)
* [Forward & Backward Passes](http://nbviewer.jupyter.org/github/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/02_fully_connected_my_reimplementation.ipynb?flush_cache=true)
    * [My Medium Post on Weight Initialization](https://medium.com/@jamesdell/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79)    
#### Papers Covered
* [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)
* [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a.html)
* [Fixup Initialization: Residual Learning Without Normalization](https://arxiv.org/abs/1901.09321)

### Week 9: How to Train Your Model
* [Mini-batch Training](http://nbviewer.jupyter.org/github/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/03_minibatch_training_my_reimplementation.ipynb?flush_cache=true)
* [Callbacks](http://nbviewer.jupyter.org/github/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/04_callbacks_my_reimplementation.ipynb?flush_cache=true)
* [Annealing](http://nbviewer.jupyter.org/github/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/05_anneal_my_reimplementation.ipynb?flush_cache=true)
#### Papers Covered
* [All you need is a good init](https://arxiv.org/abs/1511.06422)
* [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](https://arxiv.org/abs/1312.6120)
* [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)

### Week 10: 
* []()
#### Papers Covered
* []()