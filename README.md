# Setup
run `init.sh`:
- downloads python 3.9.9
- extracts it
- builds it
- creates virtualenv with it
- installs dependencies

# Run
available scrips that run experiments:
- `Run1D.sh` all 1 dimensional experiments
- `Run2D.sh` all 2 dimensional experiments
- `Run10D.sh` all 10 dimensional experiments
- `RunMiniboone.sh` perform MiniBooNE experiment

# Packages
- `aftermath`
  - prints all the nice diagrams and merges tables
  - not needed if you do not want all tables and all diagrams
- `broken`
  - stuff that broke somewhere along the line: MNIST, HepMass
- `common`
  - common stuff like serialisation
  - multiprocessing (more than one implementation, because I had to learn how to do it right) to speed things up and prevent Tensorflow from wasting memory
  - global variables/switches
- `distributions`
  - parametric distributions:
    - Gaussian
    - uniform
  - multivariate
  - multimodal
    - weighted multimodal
  - learned distributions with serialisation
- `kl` 
  - Kullback-Leibler-divergence
- `keta`
  - serialisation for tensorflow models
- `maf` - Masked Autoregressive Flow:
  - `dim1`, `dim2`, `dim10`
    - experiments with dimensions 1,2 and 10
  - `mixlearn`
    - train MAFs and the classifiers with a mixture of genuine and synthetic samples
  - `variable`
    - all about building the training plan. will be stored as `training.plan.csv`. see below
  - `bug1.py`
    - nasty Tensorflow bug
  - `DL.py` - all about datasets
    - downloading and parsing
    - save and load
    - split
    - combine data sources
  - `NoiseNormBijector.py`
    - custom layer for NFs that can add noise and normalise the input

    

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