# Medical-Info-Extraction
Extracting the information of medical text using LSTM

Note for the embedding file GoogleNews-vectors-negative300.bin, you can download from https://code.google.com/archive/p/word2vec/.

The medical text used for the program is formated as follows:

1 line for the medical sentence
1 line for test information (the location of test word in the sentence)
1 line for problem information (the location of problem in the sentence)
1 line for Tag (What's the relationship of test and problem)

The job is to get the tag with the given medical context and the information of test & problem.

Currently, a simple LSTM approach is applied with only test input. The accuracy is around 65%. More refinement of the network would be done later.

------2019.3.19 update-------
The test and problem information have been added to the model. The accuracy has been further elevated by 30 percent!
