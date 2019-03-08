# Design Notes

A few notes on why some of the (potentially counterintuitive) architectural and technical decisions in the app are made.

## Overall

* A primary design goal of automl-gs is to help build the *entire* modeling pipeline. The majority of AutoML approaches just handle the training as they are designed more for research-oriented purposes, but in the real world, that's half the battle, and sometimes the *easier* part. A goal of automl-gs is to add more production-oriented features to the generated scripts! (e.g. serving/caching)
* Yes, one of the major motivations for creating this project is to cheese Kaggle competitions. Many data science thought pieces about building models for Kaggle competition recommended similar ETL steps and architectures done automatically by automl-gs; my hope is that this package **democratizes** the process a bit more, especially with native compatability for Kaggle Kernels. As the README foreword notes, automl-gs is just a baseline that won't win a Kaggle competition on its own, but it should level the playing field.
* A core design principle is that automl-gs is not required to run the generated scripts. One issue I have with recent well-funded AI SaaS startups is their aggressive use of lock-in to keep the customer in their ecosystem, making it hard to deploy your model on cheaper hardware. In my opinion, this lock-in is not a sustainable business practice.
* I try to avoid using pretrained networks (e.g. for text/images) because they are not user-friendly and may be difficult to run on weaker hardware. Additionally, pretrained approaches assume the destination model architure is similar: for example, GloVe text vectors assume the only model input is a given text; if there are other model inputs, they won't be effective.
* A newer trend in AutoML is neural architecture search (NAS), as implemented in adanet and auto-keras. However, NAS has a [much lower ROI](https://www.pyimagesearch.com/2019/01/07/auto-keras-and-automl-a-getting-started-guide/) for algorithm quality vs. training time, which makes it much less accessible for people with smaller budgets. As a result, implementing it into automl-gs is not an immediate priority.
* Model weights and encoders are not encoded as pickle files since pickled filed expose security risks and may be rendered invalid by later Python versions.
* Discrete hyperparameter values are used instead of uniform hyperparamerters to make comparing across a given value more consistent.

## ETL Pipeline

* The auto-column type inference is still a work in progress, as there are no consistent heuristics on how to do it accurately; hence why there is an option provide column types explicitly if necessary. Feel free to file an issue if there is a hole in the heuristics.
* `id`/`uuid`/`guid` fields are iignored since there is zero statistical insight to gain from their values (if the order when rows are created is important, a `created_at` Datetime field will capture that).
* `dayofweek` and `hour` are *always* extracted for datetime fields. `month` and `year` have a possiblity of overfitting in combination with `dayofweek`/`hour`, which is why they are tuned.
* Multiple input text fields will always use the same encoder / shared model architecture for efficiency.
* The `quantiles` and `percentiles` strategies for bucketing numeric data are an improvement on the traditional break-data-into-*n*-equal-intervals as it ensures a healthy amount of each input class; numeric data is rarely so uniform.

## Data Analysis

* In addition to the targeted performance metric, all common metrics for the given problem type are also collected. This may sound excessive, but it's much better to be safe than sorry, and the extra metrics can be used as a sanity check.

## Model Training

* TensorFlow regression problems use a hyperparameter trying to optimize different metrics to see which one works best: `mse` (mean square error; the typical metric for regression problems), `msle` (mean square logarithmic error, for massively-skewed targets), and `poisson` (poisson loss, for fields which can be modeled as a sum of indepenent events). Optimizing for `msle`/`poisson` can have better results for `mse` if it follows into the data distributions above even if not directly optimizing for `mse`. (for example, by incentivising the MLP to engineer the appropriate inverse function)
* A custom logger is used to ensure all metrics are recorded properly for all frameworks.
* The logger also stores the timestamp when each epoch finishes; this provides a log both of overall training, and allows you to derive per-epoch training time if necessary.

## TensorFlow (tf.keras)

* Keras is used instead of base TensorFlow to simplify further development on the model.
* The functional Keras API is used instead of the sequential API to support model multi-input into the MLP to merge everything together.
* tf.keras is used instead of external Keras due to better compatability with TensorFlow-only features (Optimizers, Datasets, TPUs, etc.). This may change after TensorFlow 2.0.
* The AdamW optimizer + Cosine Annealing w/ Warm Restarts method is used due to the good results from the paper [Fixing Weight Decay Regularization in Adam](https://arxiv.org/abs/1711.05101) granting that configuration robustness in the hyperparameter selection space.
* The model is built at runtime, and only the weights of the model are saved. This is different that the TensorFlow SavedModel approach, but this approach allows for models to be built with more complex and conditional architectures (e.g. for text, use a normal LSTM on CPU-only platforms but a CuDNNLSTM when a GPU is present).