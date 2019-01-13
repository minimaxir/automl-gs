# Design

A few notes on why some of the architecture decisions in the app are made.

## Overall

* Yes, one of the major motivations for creating this project is to cheese Kaggle competitions. Many data science thought pieces about building models for Kaggle competition recommended similar ETL steps and architectures done automatically by automl-gs; my hope is that this package **democratizes** the process a bit more, especially with native compatability for Kaggle Kernels. As the README foreword notes, automl-gs is just a baseline that won't win a Kaggle competition on its own, but it should level the playing field.
* A core design principle is that automl-gs is not required to run the generated scripts. One issue I have with recent well-funded AI startups is their aggressive use of lock-in to keep the customer in their ecosystem, making it hard to deploy your model on cheaper hardware. In my opinion, this lock-in is not a sustainable business practice.
* I try to avoid using pretrained networks (e.g. for text/images) because they are not user-friendly and may be difficult to run on weaker hardware. Additionally, pretrained approaches assume the destination model architure is similar: for example, GloVe text vectors assume the only model input is a given text; if there are other model inputs, they won't be effective.
* A newer trend in AutoML is neural architecture search (NAS), as implemented in adanet and auto-keras. However, NAS has a [much lower ROI](https://www.pyimagesearch.com/2019/01/07/auto-keras-and-automl-a-getting-started-guide/) for algorithm quality vs. training time, which makes it much less accessible for people with smaller budgets. As a result, implementing it into automl-gs is not an immediate priority.

## ETL Pipeline

* `id`/`uuid` fields are iignored since there is zero statistical insight to gain from their values (if the order when rows are created is important, a `created_at` Datetime field will capture that).
* However, foreign relationships (e.g. fields suffixed with `_id` ) might be important; this is a special case where such a field is always encoded as categorical instead of numeric.
* `dayofweek` and `hour` are *always* used for datetime fields. `month` and `year` have a possiblity of overfitting in combination with `dayofweek`/`hour`, which is why they are tuned.

## Model Training

* TensorFlow regression problems use a hyperparameter trying to optimize different metrics to see which one works best: `mse` (mean square error; the typical metric for regression problems), `msle` (mean square logarithmic error, for massively-skewed targets), and `poisson` (poisson loss, for fields which can be modeled as a sum of indepenent events). Optimizing for `msle`/`poisson` can have better results for `mse` if it follows into the data distributions above even if not directly optimizing for `mse`. (for example, by incentivising the MLP to engineer the appropriate inverse function)

## TensorFlow (tf.keras)

* Keras is used instead of base TensorFlow to simplify further development on the model.
* The functional Keras API is used instead of the sequential API to support model multi-input into the MLP to merge everything together.
* tf.keras is used instead of external Keras due to better compatability with TensorFlow-only features (Optimizers, Datasets, TPUs, etc.). This may change after TensorFlow 2.0.
* The AdamW optimizer + Cosine Annealing w/ Warm Restarts method is used due to the good results from the paper [Fixing Weight Decay Regularization in Adam](https://arxiv.org/abs/1711.05101) granting that configuration robustness in the hyperparameter selection space.