# Medical-Info-Extraction
This repository contains several approaches of entity relation extraction, with regard to dataset i2b2, which i uploaded in the repository.

Two main approaches are used and compared, which are
* LSTM + attention mechanism
* BERT, which is state-of-the-art NLP model

## LSTM
Several LSTM networks are tried. Since the dataset is relatively small and has a lot of repetitions, the model should not be too complex.
Currently the best LSTM model for the task is as follows:
![Image text](https://github.com/Ledzy/Medical-Info-Extraction/blob/master/LSTM_experiments/图片1.png)

which references: 
[Zhou, Peng, et al. "Attention-based bidirectional long short-term memory networks for relation classification." Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). Vol. 2. 2016.](https://www.aclweb.org/anthology/P16-2034)

Note for the embedding file GoogleNews-vectors-negative300.bin, you can download from https://code.google.com/archive/p/word2vec/.

## BERT
To be added...
