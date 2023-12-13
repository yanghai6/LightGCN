# LightGNN

We enhanced the LightGCN model for recommender system by introducing two methods of negative sampling. 

**Batch-Level Negative Sampling**
To execute hard negative sampling on the batch-level, we train the model using uniformly random sampled negative examples on the first batch. We then utilize the trained model to score all possible training examples in the previous batch. The negative examples receiving the highest scores from the model are deemed as the hard negative examples. Subsequently, we integrate these hard negative examples with the set of positive samples in the next batch to calculate the loss. 

**Epoch-Level Negative Sampling**
We take a more global approach for epoch-level negative sampling. For the initial epoch, we use uniform negative sampling for negative examples. From the second epoch onward, we compute the scores for all negative samples across all users in the previous epoch, and select those with the highest scores. By incorporating these epoch-level hard negative examples with the positive samples in the subsequent training, we aim to provide the model with challenging instances that span the entire dataset.

**Usage**
The 4 python notebooks are used to examine the reliability of the negative sampling methods. Each notebook includes three distinct runs of one sampling method on one datasets, with comparison with the baseline model (LightGCN without hard negative sampling).

- LightGCN_Hard_Batch_Amazon_Book: batch-level negative sampling on Amazon Books Dataset
- LightGCN_Hard_Batch_MovieLens: batch-level negative sampling on MovieLens Dataset
- LightGCN_Hard_Epoch_Amazon_Book: epoch-level negative sampling on Amazon Books Dataset
- LightGCN_Hard_Epoch_MovieLens: epoch-level negative sampling on MovieLens Dataset
