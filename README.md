# Self-Training for Sample-Efficient Active Learning for Text Classification with Pre-Trained Language Models

Experiment code for the paper [Self-Training for Sample-Efficient Active Learning for Text Classification with Pre-Trained Language Models](https://arxiv.org/pdf/2406.09206).    
Accepted at EMNLP 2024 Main.

---

## Requirements

- Python 3.8+  
- CUDA-capable GPU

---

## Installation

To install the required dependencies, run:

```bash
pip -r requirements.txt
```

Note: For the sake of any easier installation, we included `faiss-gpu` as a dependency here. 
In our experiments, and depending on your CUDA version and system, you might need to compile faiss yourself.

For exact versions used in the experiment, refer to requirements-frozen.txt.

## Initial Steps

After installing dependencies, two steps have to performed before you can start.

### Set Up Mlflow logging

This application uses [mlflow](https://www.mlflow.org/) to log results. 
Before the first run you must create a mlflow experiment:

```bash
# replace [NEW_EXPERIMENT_NAME] by some string for example by new_experiment
mlflow experiments create -n [NEW_EXPERIMENT_NAME]
```

On creation, you will be shown the experiment ID (which you could also look up later). 
The  results will later be written to `mlruns/[experiment ID]`.

### Generate the DBP-140k Dataset

To create the subsampled DBPedia dataset, run `data/create_dbp140k.py` from the main directory:

```bash
python data/create_dbp140k.py
```

---

## Usage

The working dir is expected to be the top-level directory of this project. (You could change this but then the following commands might not work without alterations.)

### General

The general syntax is as follows:

```bash
python -m active_learning_lab.experiments.active_learning.active_learning_runner [config_file] [arguments]
```

where config file points to a python file (in python module syntax, i.e. `path.to.config_file`). 
This config file also defines the mlflow experiment to use, so make sure that the respective experiment exists.


To view all possible arguments for the runner, use `-h`:

```bash
python -m active_learning_lab.experiments.active_learning.active_learning_runner -h
```

---

### Quick Start for a Self-Training Experiment

For a self-training experiment, you need to specify a dataset, a classifier, a query strategy, and a self-training strategy. This can be done as follows:

```bash
python -m active_learning_lab.experiments.active_learning.active_learning_runner active_learning_lab.config.active_learning.self_training.arr_2024 \
--dataset_name ag-news \
--classifier_name transformer \
--query_strategy random \
--active_learner self-training \
--active_learner_kwargs self_training_method=hast
```

The config file used in this example is the one used for the main experiments and shows details about the experiments.

### Possible Values

| Parameter | Values                          | 
| --------- |---------------------------------|
| dataset_name | ag-news, dbp-140k, imdb, trec   |
| classifier_name | transformer, setfit-ext         | 
| query_strategy | lc-bt, cal, random              | 
| self_training_method | ust, actune, verips, nest, hast |


`classifier_name=transformer` equals BERT in this case and setfit-ext is SetFit.

### Inspecting the Results

The results are similar to a predecessor of this project (v1), for which we described the results at [webis-de/acl22-revisiting-uncertainty-based-query-strategies-for-active-learning-with-transformers](https://github.com/webis-de/acl22-revisiting-uncertainty-based-query-strategies-for-active-learning-with-transformers/blob/main/USAGE.md#inspecting-the-results) where the results are described as well.

---

### Limitations

While the experiments have been extensively tested with `self_training_iterations=1` (which is the setting used in the paper), 
for multiple self-training iterations there is a remaining TODO note in `active_learning_lab/experiments/active_learning/self_training/strategies/ust.py` regarding support for multiple self-training iterations.

If you plan to use multiple iterations, please verify the implementation details in ust.py to ensure compatibility, regardless of the self-training strategy chosen.

---

### License

Under `active_learning_lab/thirdparty`, there is some thirdparty code, which was adopted and then adapted to extend the original functionality.


- SentenceTransformers: [Apache 2.0](https://github.com/UKPLab/sentence-transformers/blob/v2.2.2/LICENSE)
- SetFit: [Apache 2.0](https://github.com/huggingface/setfit/blob/v0.7.0/LICENSE)

All other code is licensed under the [MIT License](License).

See the [LICENSE](LICENSE) and [LICENSE-THIRDPARTY](LICENSE-THIRDPARTY) files for the full licenses.

---

### Acknowledgments

The authors acknowledge the financial support by the Federal Ministry of Education and Research of Germany and by Sächsische Staatsministerium für Wissenschaft, Kultur und Tourismus in the programme Center of Excellence for AI-research "[Center for Scalable Data Analytics and Artificial Intelligence Dresden/Leipzig](https://scads.ai/)", project identification number: ScaDS.AI.

We would like to thank the [Webis Group](https://webis.de/) and the [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/) for providing GPU resources.
We are especially grateful to the anonymous reviewers for their highly constructive and valuable feedback.
