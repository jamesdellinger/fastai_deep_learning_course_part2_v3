# [Fast.ai](https://course.fast.ai/part2) Deep Learning from the Foundations (Spring 2019)
*[Part II](https://www.youtube.com/watch?v=4u8FxNEDUeg) of Fast.ai's two-part deep learning course, offered through [The Data Institute at USF](https://www.usfca.edu/data-institute/certificates/deep-learning-part-two). From March through the end of April in 2019. My Part I coursework is [here](https://github.com/jamesdellinger/fastai_deep_learning_course_part1_v3).*

This course offered a bottom-up approach (through code, not math equations) to becoming an expert deep learning practitioner and experimenter. 

We implemented core [fastai](https://github.com/fastai/fastai) and [PyTorch](https://pytorch.org/docs/stable/index.html) classes and modules from scratch, achieving similar or better performance. We also practiced coding up techniques introduced in various papers, and then spent significant time on strategies useful in decreasing model training time (parallelization, JIT). The final two weeks were spent diving deep into [Swift for TensorFlow](https://www.tensorflow.org/swift) with [Chris Lattner](http://www.nondot.org/sabre/), where we saw first-hand how [differentiable programming](https://medium.com/@karpathy/software-2-0-a64152b37c35) could work.

I came away with both the know-how to engineer cutting-edge deep learning ideas from scratch with optimized code, as well as the expertise necessary to research and explore new ideas of my own.

## My Explanations of Lesson Notebooks
I implemented the code created by [Jeremy Howard](https://www.fast.ai/about/#jeremy) and [Sylvain Gugger](https://www.fast.ai/about/#sylvain) for the course's weekly lectures and reproduced their results. However, the bulk of my time (about 5 months, full-time study) was spent crafting my own plain-English explanations of the techniques and concepts that the class covers. I'm proudest of my explorations of [language model pre-training](http://nbviewer.jupyter.org/github/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/12_text_my_reimplementation.ipynb?flush_cache=true) and the [Swift for TensorFlow framework](http://nbviewer.jupyter.org/github/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/14_swift_autodiff_explanation.ipynb?flush_cache=true).

### Week 8: Building Optimized Matmul, Forward and Backpropagation from Scratch
* [Matrix Multiplication](http://nbviewer.jupyter.org/github/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/01_matmul_my_reimplementation.ipynb?flush_cache=true)
* [Forward & Backward Passes](http://nbviewer.jupyter.org/github/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/02_fully_connected_my_reimplementation.ipynb?flush_cache=true)
    * [My Medium Post on Weight Initialization](https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79)    

### Week 9: How to Train Your Model
* [Mini-batch Training](http://nbviewer.jupyter.org/github/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/03_minibatch_training_my_reimplementation.ipynb?flush_cache=true)
* [Callbacks](http://nbviewer.jupyter.org/github/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/04_callbacks_my_reimplementation.ipynb?flush_cache=true)
* [Annealing](http://nbviewer.jupyter.org/github/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/05_anneal_my_reimplementation.ipynb?flush_cache=true)

### Week 10: Wrapping up CNNs
* [Early Stopping](http://nbviewer.jupyter.org/github/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/05b_early_stopping_my_reimplementation.ipynb?flush_cache=true)
* [CUDA Training and Hooks](http://nbviewer.jupyter.org/github/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/06_cuda_cnn_hooks_init_my_reimplementation.ipynb?flush_cache=true)
* [Batchnorm](http://nbviewer.jupyter.org/github/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/07_batchnorm_my_reimplementation.ipynb?flush_cache=true)

### Week 11: Data Loading, Optimizers, and Augmentations 
* [Layer-Sequential Unit-Variance (LSUV) Weight Initialization](http://nbviewer.jupyter.org/github/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/07a_lsuv_my_reimplementation.ipynb?flush_cache=true)
* [Building fastai's DataBlock API from Scratch](http://nbviewer.jupyter.org/github/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/08_data_block_my_reimplementation.ipynb?flush_cache=true)
* [Improving PyTorch's Optimizers](http://nbviewer.jupyter.org/github/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/09_optimizers_my_reimplementation.ipynb?flush_cache=true)
* [Image Augmentation and PyTorch JIT](http://nbviewer.jupyter.org/github/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/10_augmentation_my_reimplementation.ipynb?flush_cache=true)
* [NVIDIA's DALI Batch Image Augmentation Library](http://nbviewer.jupyter.org/github/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/Diving_into_DALI.ipynb?flush_cache=true)
    * [My Medium Post on DALI](https://towardsdatascience.com/diving-into-dali-1c30c28731c0) 

### Week 12: MixUp, XResNets, AWD-LSTM and ULMFiT
* [Mixup and Label Smoothing](http://nbviewer.jupyter.org/github/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/10b_mixup_label_smoothing_my_reimplementation.ipynb?flush_cache=true)
* [FP16 Training](http://nbviewer.jupyter.org/github/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/10c_fp16_my_reimplementation.ipynb?flush_cache=true)
* [A Flexible & Concise XResNet Implementation](http://nbviewer.jupyter.org/github/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/11_train_imagenette_my_reimplementation.ipynb?flush_cache=true)
* [Transfer Learning from Scratch](http://nbviewer.jupyter.org/github/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/11a_transfer_learning_my_reimplementation.ipynb?flush_cache=true)
* [A Survey of Language Model Techniques](http://nbviewer.jupyter.org/github/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/12_text_my_reimplementation.ipynb?flush_cache=true)

### Weeks 13 and 14: Implementing fastai/PyTorch classes from scratch in Swift!
* [Making an XResNet34 Training Loop](http://nbviewer.jupyter.org/github/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/13_swift_resnet_pipeline_s4tf_v04_my_reimplementation.ipynb?flush_cache=true)
* [Explaining Automatic Differentiation with Swift](http://nbviewer.jupyter.org/github/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/14_swift_autodiff_explanation.ipynb?flush_cache=true)

## Papers We Studied
### Week 8
* [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)
* [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a.html)
* [Fixup Initialization: Residual Learning Without Normalization](https://arxiv.org/abs/1901.09321)
### Week 9
* [All you need is a good init](https://arxiv.org/abs/1511.06422)
* [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](https://arxiv.org/abs/1312.6120)
* [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
### Week 10
* [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
* [Layer Normalization](https://arxiv.org/abs/1607.06450)
* [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
* [Group Normalization](https://arxiv.org/abs/1803.08494)
* [Revisiting Small Batch Training for Deep Neural Networks](https://arxiv.org/abs/1804.07612)
### Week 11
* [All you need is a good init](https://arxiv.org/abs/1511.06422)
* [Decoupled Weight Regularization](https://arxiv.org/abs/1711.05101.pdf)
* [L2 Regularization versus Batch and Weight Normalization](https://arxiv.org/abs/1706.05350)
* [Norm matters: efficient and accurate normalization schemes in deep networks](https://arxiv.org/abs/1803.01814)
* [Three Mechanisms of Weight Decay Regularization](https://arxiv.org/abs/1810.12281)
* [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
* [Reducing BERT Pre-Training Time from 3 Days to 76 Minutes (LAMB optimizer paper)](https://arxiv.org/abs/1904.00962)
* [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
### Week 12
* [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
* [Rethinking the Inception Architecture for Computer Vision (label smoothing is in part 7)](https://arxiv.org/abs/1512.00567)
* [Bag of Tricks for Image Classification with Convolutional Neural Networks (XResNets)](https://arxiv.org/abs/1812.01187)
* [Regularizing and Optimizing LSTM Language Models (AWD-LSTM)](https://arxiv.org/abs/1708.02182)
* [Universal Language Model Fine-tuning for Text Classification (ULMFiT)](https://arxiv.org/abs/1801.06146)

## Dependencies for PyTorch Notebooks
* [requirements.txt](https://github.com/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/requirements_pytorch.txt)
* [environment.yml](https://github.com/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/pytorch.yml)

## USF Completion Certificate
[<img src="https://github.com/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/images/USF_completion_cert_page1.jpg" height="250">](https://github.com/jamesdellinger/fastai_deep_learning_course_part2_v3/blob/master/USF_completion_cert_page1.pdf)