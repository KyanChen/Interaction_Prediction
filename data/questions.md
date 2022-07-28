1. What’s your prediction outcome variable?

I divide the 13 interaction patterns into 13 distinct categories, 
the predicted output is the probability that the protein-protein interaction belongs to one of the 13 patterns,
and the highest pattern is chosen as the output.

2. How’s the training cohort constructed?

Using deep learning methods to predict protein-protein interaction,
the network input is the code id of two proteins (each protein is assigned a unique code), 
the output is the probability of belonging to one of 13 interaction modes, 
and 1024 pieces of data are combined into one minibatch, with AdamW optimizer for optimization.

3. What are the features used and how are they constructed?

Using the code id of the two proteins as the feature input, 
each protein is treated as a word, encoded into the embedding space (256 dims), 
and then the cross-attention algorithm is used to learn the two proteins' interaction pattern. 
Because this interaction is order agnostic, no position encoding is used.

4. How do you evaluate the model?

I separated the all data into a training set and a test set (1w pieces of data), 
obtained the model parameters on the training set, 
and tested the model's accuracy on the test set.
Experiments show that the method can achieve 88% accuracy on the test set without optimizing the network structure.

5. What would be the next steps?

Add protein-to-protein co-occurrence priors; Augment the data; Optimize the network structure; Consider the balance of interaction modes
