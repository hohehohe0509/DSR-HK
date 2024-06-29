# DSR-HK
Session-based recommendation systems need to capture implicit user intents from sessions, but existing models suffer from issues like item interaction dominance, noisy knowledge graphs, and session noise. We propose a multi-channel model with a knowledge graph channel, session hypergraph channel, and session line graph channel to capture relevant information from knowledge graphs, within sessions, and beyond sessions respectively. In the knowledge graph channel, we adapt ively remove redundant edges to reduce noise. Knowledge graph representations cooperate with hypergraph representations for prediction to alleviate item dominance. We also generate in-session attention for denoising. Finally, we maximize mutual information between the hypergraph and line graph channels as an auxiliary task.\\

![image](https://github.com/hohehohe0509/DSR-HK/blob/main/%E6%9E%B6%E6%A7%8BV1.PNG)

## Datasets
We use three datasets to test the model. Here is an introduction to these datasets:
1. [Tmall](https://tianchi.aliyun.com/dataset/140281): The Tmall dataset is a widely used resource in recommendation system research, sourced from Alibaba's e-commerce platform. This dataset includes millions of user-item interaction records spanning several months, capturing temporal patterns in user behavior. It encompasses user IDs, item IDs, brands, categories, user actions (such as clicks, favorites, add-to-cart, and purchases), and timestamps. To protect user privacy, all IDs are anonymized.

2. [Retailrocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset): The Retailrocket dataset originates from a real e-commerce company, containing user behavior data over a six-month period. It includes records of user views, add-to-cart actions, and transactions, along with product metadata. This dataset is popular among recommendation system researchers due to its authenticity and rich user behavior information, making it particularly suitable for developing and testing personalized recommendation algorithms.

3. KKBOX: The KKBOX dataset is a music recommendation dataset provided by KKcompany for a private Kaggle competition. It records users' music listening sequences and includes metadata information for each song. To protect user privacy, all data has been hashed. In this dataset, each session is set to a length of 20 songs, and each session has 5 ground truths. In our experiment, we only use the first one as the ground truth.


## Requirements
The entire experiment is running in an Ubuntu environment with Cuda version 11.8. 
Please download and install the following packages to configure the environment:
```
#cuda 11.8

pip install --user -U scikit-learn==0.23.2
pip install python 3.10
pip install pytorch 2.0.1+cu118
pip install numpy 1.26.2
pip install scipy 1.11.4
pip install numba 0.58.1
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
```

## File Description
The "datasets/" folder contains three preprocessed datasets used for testing the experiment: Tmall, Retailrocket, and KKBOX. Each dataset has its own folder named after itself, containing the following files:
* kg.txt: This is the knowledge graph file. Each line represents a triple in the format "entity relation entity".
* all_train_seq.txt: This is the original training set before session splitting. The content has been converted using pickle. The input is a sequence of item IDs, and the target is a single item ID.
* train.txt: This is the training set after session splitting. The content has been converted using pickle. The input is a sequence of item IDs, and the target is a single item ID.
* test.txt: This is the test set after session splitting. The content has been converted using pickle. The input is a sequence of item IDs, and the target is a single item ID.

The "DSR-HK/" folder contains program files for the entire experiment and model architecture. It includes four files:
* con.py: This file details the module responsible for handling hypergraph convolution in the model architecture.
* main.py: This file manages the model's hyperparameters and declarations.
* model.py: This file contains the main structure of the model.
* util.py: This file is responsible for processing the structure of datasets to be input into the model.

## Execution
Please run the code as follow:
```
python main.py [Options]
```


### Options
| Name                 | Default           | Description                                                         |
|:-------------------- | ----------------- |:------------------------------------------------------------------- |
| `--dataset`          | `Retailrocket`    | Dataset name </br> Options: `Tmall`, `KKBOX` or `Retailrocket`      |
| `--epoch`            | `7`               | Number of epochs to train for                                       |
| `--batchSize`        | `100`             | Knowledge graph training batch size                                 |
| `--kg_batch_size`    | `100`             | Recommendation training batch size                                  |
| `--embSize`          | `112`             | Item embedding size                                                 |
| `--kg_embSize`       | `112`             | Entity embedding size                                               |
| `--relation_embSize` | `112`             | Relation embedding size                                             |
| `--lr`               | `0.001`           | Learning rate                                                       |
| `--seed`             | `-2023`           | Random seed used in the experiment.                                 |
| `--layer`            | `1`               | The number of layer used                                            |
| `--beta`             | `0.001`           | ssl task maginitude                                                 |
| `--layer_size`       | `[112, 112, 112]` | Output size of each layer in the knowledge graph convolution        |
| `--adj_type`         | `si`              | Specify the type of the adjacency (laplacian) matrix from {bi, si}. |
| `--K`                | `5`               | Number of positive and negative samples                             |
| `--temperature`      | `0.1`             | Temperature parameter of contrastive loss                           |
### Example
```bash
$ python main.py \
  --dataset="Tmall" --epoch=10  \
  --layer=2  \
  --beta=0.05
```

To obtain the results reported in the paper, please set the parameters as follows:
|             | Tmall | KKBOX | Retailrocket |
| ----------- | ----- | ----- | ------------ |
| layer       | 1     | 1     | 1            |
| beta        | 0.05  | 0.001 | 0.001        |
| temperature | 0.1   | 0.05  | 1            |
| K           | 5     | 5     | 5            |

Other parameters do not need to be changed.
