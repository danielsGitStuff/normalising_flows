# Setup
run `init.sh`:
- downloads python 3.9.9
- extracts it
- builds it
- creates virtualenv with it
- installs dependencies


# Training Plan columns
- done `[0, 1]` 1 means a classifier has been trainied and evaluated
- dsize size of the whole training data set (for the MAF)
- synthratio `[0.0 .. 1.0]` ratio of genuine vs synthetic data for training the classifier, 0.0 means 100% genuine data 
- model `[0 .. inf]` model id for a certain configuration, training multiple models per configuration is possible
- clfsize `[1 .. inf]` amount of training samples for classifier
- tsize `[1 .. inf]` amount of training samples for the MAF
- vsize `[0 .. inf]` amount of val samples for the MAF, used for EarlyStopping
- clf_t_g_size `[0 .. inf]` amount of genuine training samples for training the classifier
- clf_t_s_size `[0 .. inf]` amount of synthetic training samples for training the classifier
- clf_v_g_size `[0 .. inf]` amount of genuine val samples for training the classifier
- clf_v_s_size `[0 .. inf]` amount of synthetic val samples for training the classifier
- loss `[-inf .. inf]` metric: loss on test data
- accuracy `[0.0 .. 1.0]` metric: accuracy on test data
- max_epoch `[0 .. inf]` metric: epoch the classifier stopped training
- tnoise `[0 .. inf]` metric: True Noise
- fnoise `[0 .. inf]` metric: False Noise
- tsig `[0 .. inf]` metric: True Signal
- fsig `[0 .. inf]` metric: False Signal