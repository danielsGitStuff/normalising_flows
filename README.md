# Contents
- [Setup](#setup)
- [Run](#run)
- [Packages](#packages)
- [Documents](#documents)
- [Training plan columns](#training-plan-columns)
- [Possible traps to avoid](#what-traps-can-you-avoid-when-you-write-a-thesis)


# Setup
run [`init.sh`](init.sh):
- downloads python 3.9.9
- extracts it
- builds it
- creates virtualenv with it
- installs dependencies

# Run
available scrips that run experiments:
- [`Run1D.sh`](Run1D.sh) all 1 dimensional experiments
- [`Run2D.sh`](Run2D.sh) all 2 dimensional experiments
- [`Run10D.sh`](Run10D.sh) all 10 dimensional experiments
- [`RunMiniboone.sh`](RunMiniboone.sh) perform MiniBooNE experiment

# Packages
- [`aftermath`](aftermath/)
  - prints all the nice diagrams and merges tables
  - not needed if you do not want all tables and all diagrams
- [`broken`](broken/)
  - stuff that (likely but not necessarily) broke somewhere along the line: MNIST, HepMass
- [`common`](common/)
  - common stuff like serialisation
  - multiprocessing (more than one implementation, because I had to learn how to do it right) to speed things up and prevent Tensorflow from wasting memory
  - global variables/switches
- [`distributions`](distributions/)
  - parametric distributions:
    - Gaussian
    - uniform
  - multivariate
  - multimodal
    - weighted multimodal
  - learned distributions with serialisation
  - [`kl` ](distributions/kl/)
    - Kullback-Leibler-divergence
- [`doc`](doc/)
  - Thesis and presentation live here
- [`keta`](keta/)
  - serialisation for tensorflow models
- [`maf`](maf/) - Masked Autoregressive Flow:
  - [`dim1`](maf/dim1/), [`dim2`](maf/dim2/), [`dim10`](maf/dim10/)
    - experiments with dimensions 1,2 and 10
  - [`mixlearn`](maf/mixlearn/)
    - train MAFs and the classifiers with a mixture of genuine and synthetic samples
  - [`variable`](maf/variable/)
    - all about building the training plan. will be stored as `training.plan.csv`. see below
  - [`bug1.py`](maf/bug1.py)
    - nasty Tensorflow bug
  - [`DL.py`](maf/DL.py) - all about datasets
    - downloading and parsing
    - save and load
    - split
    - combine data sources
  - [`NoiseNormBijector.py`](maf/NoiseNormBijector.py)
    - custom layer for NFs that can add noise and normalise the input

# Documents
- [Thesis](doc/master_nf.pdf)
- [Presentation](doc/presentation/presentation%20final.pdf)
- [Presentation notes](doc/presentation/presentation%20final%20notes.odt)

    

# Training Plan Columns
This are the columns in the `training.plan.csv` you will find in `.cache/mixlearn_{some name}/`.
The file contains all information about what amount of data from what dataset (training/val/test) went into which classifier what performance it achieved.

- done `[0, 1]` 1 means a classifier has been trainied and evaluated
- model `[0 .. inf]` model id for a certain configuration, training multiple models per configuration is possible
- clf_t_ge_noi `[0 .. inf]` amount of genuine training samples of noise for training a classifier
- clf_t_ge_sig `[0 .. inf]` amount of genuine training samples of signal for training a classifier
- clf_t_sy_noi `[0 .. inf]`
- clf_t_sy_sig `[0 .. inf]` amount of synthetic training samples of signal for training a classifier
- clf_v_ge_noi `[0 .. inf]`
- clf_v_ge_sig `[0 .. inf]` amount of genuine validation samples of signal for training a classifier
- clf_v_sy_noi `[0 .. inf]`
- clf_v_sy_sig `[0 .. inf]`
- clfsize `[1 .. inf]` total amount of training samples for a classifier
- dsize `[1 .. inf]` size of the whole training dataset for the MAF
- size_clf_t_ge `[0 .. inf]` total amount of genuine training samples for training a classifier
- size_clf_t_sy `[0 .. inf]` total amount of synthetic training samples for training a classifier
- size_clf_v_ge `[0 .. inf]`
- size_clf_v_sy `[0 .. inf]`
- size_nf_t_noi `[0 .. inf]` total amount of noise training samples for training the MAF
- size_nf_t_sig `[0 .. inf]` total amount of signal training samples for training the MAF
- size_nf_v_noi `[0 .. inf]` total amount of noise validation samples for training the MAF
- size_nf_v_sig `[0 .. inf]`
- test_clf_no `[0 .. inf]` total amount of noise test samples for the classifiers
- test_clf_sig `[0 .. inf]` total amount of signal test samples for the classifiers
- tsize `[1 .. inf]` size of the training dataset for the MAF
- accuracy `[0 .. 1.0]` metric: accuracy on test data
- fnoise `[0 .. inf]` metric: True Noise
- fsig `[0 .. inf]` metric: False Signal
- loss `[0 .. inf]` metric: loss on test data
- max_epoch `[0 .. inf]` metric: epoch the classifier stopped training
- tnoise `[0 .. inf]` metric: False Noise
- tsig `[0 .. inf]` metric: True Signal


# What traps can you avoid when you write a thesis?
Or things that went wrong... mostly regarding software development!

**TL;DR**
- do simple experiments first
- after that:
  - build a giant table that determines every aspect of the variable configuration parameters of you experiments ... `training.plan.csv` comes here
  - DO NOT take any intermediary steps, like 'I am varying this parameter' and adapt your training procedure and evaluation procedures to it. Go the full way.


**Long and tedious**

The biggest point I want to make is not to underestimate software complexity when it comes to build prototypes.
I first started with simple model that basically had one or more fixed output files (`.csv`s, `.json`s and what not)
and that evolved to something handling 1000 classifiers and their results. Ensuring everything gets exactly the data it must have an nothing more and putting the outputs into meaningful places became THE major thing. That is when I came up with the training plan, which is a congregation of a lot of experiments conducted.
It also includes the exact amount of samples from the different sources (genuine or synthetic and noise or signal) which makes handling this quite easy. If you have it!
My advise is, if you want to do anything beyond, let's say 3 experiments, build something like the training plan.
This way you can stupidly iterate over its rows and configure every single experiment easily. If you write the results back you can easily analyse the results and print it. YOU WANT THAT!

You have to start somewhere and when you start working on something and start to understand it more deeply you cannot come up with that very satisfying solution you eventually will come up with a couple of months. So starting simple is mandatory to develop your own deep understanding of the thing you are working on. But starting simple and iteratively making it more complex to fit you needs it very time consuming. Avoid as many intermediary steps as possible and try to implement a training plan as soon as possible as it will benefit you all the time later. You will have more time digging into other things than software development issues, like comparisons of your thing vs another, what if I throw this at another domain etc.

If not you will find yourself adapting your training procedure, because you had a parameter fixed, but now you want to vary it. But because that on its own is pointless you must adapt your evaluation procedures as well. And there might be many of them and things can go wrong adapting them. **Varying new parameters should not become more complicated than changing a few strings.**

That is why there is no comparison of other techniques, like GMMs, in here. Being busy I ditched other approaches early in the work and kind of swept them under the rug.
Later on that was not even in my mind anymore.