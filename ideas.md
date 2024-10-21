Synthetic data
- GAN with RNNs making up generators
- Input is original unmasked and untransformed data
- Output is synthetic data that retains statistical accuracy but does not reveal actual values of transformed fields
- Important for sharing data while not revealing identifiable information
- Does not deal with small ns revealing information
- Nowadays we hear about synthetic data to feed into ML models for transfer learning, but there are other uses
- Protects confidentiality
- Synthetic data mimics real data
- synthetic data is more complex than dummy or mock data
- synthetic data is different from dummy or mock data
- use case of dummy or mock data
- why use GANs instead of LLMs or GPT tech -- LLMs are nutso to build in house from scratch
- GAN vs RNN? Maybe a straight RNN trained on the real data is good enough?
- ML models vs non-ML models (and next few bullet points)
- Why not simpler methods for anonymized raw data? -- pii redaction -- often can still be rebuilt to identify original person
- Parametric auto regressive models -- not great fidelity, but good anonymity and privacy
- GANs learn relationships with limited prior assumptions about the data

Issues
- We are building something that purports to safely handle PII, and that is a big risk
- If the model is not good enough in terms of anonymization, then that is a big risk
- How much testing do we need to do on the synthetic data set to confirm it is statistically accurate
- Are we training a model that can create synthetic data generally or for a scoped use case
- How would this scale, and what would performance look like
- Stretching my limits in trying to build something from scratch-ish despite its complexity and using theory
- Will likely need some form of regularization to simplify the model(s)
- How to evaluate performance? 
- Mode collapse -- the generator produces a small number of outputs that can trick the discriminator, so the outputs collapse to the few output samples
- WGAN speaks to vanishing gradients where the discriminator outperforms the generator in such a way that the generator cannot learn, so the generator's gradient vanishes. WGAN modifies the loss function to compare the real and synthetic distributions and tries to stabilize training.
- Training stability for GANs is a key issue
- Mention wide range of models to look at, even just within GANs or other generative approaches. We need a way to limit this for the scope of the project and to build a solid foundation.
- Skipping transformer models
- Discrete and categorical values cause issues because we cannot backpropagate gradients for non-continuous values from the discriminator back to the generator
- DCGAN to address mode collapse
- If we have all continuous features, we could in theory determine the distribution for each and sample from those. Discrete features give us problems because of zero gradients in backpropagation.
- Note that there is precedent for using trees and Bayesian networks. We will focus on GANs.
- CNNs vs RNNs in two source papers
- GANs for tabular data are hard because of the different data types (source paper 2 pg 2)
- We can start by looking at generating data for one feature, but we will need to expand to generating data for entire rows. This means we need a way to treat rows in their entirety as inputs and outputs.
- We may want to explore rows as sequential data with one feature following another. I am skeptical of this as of now, but it would open up using recurrent neural networks (RNNs) as GAN generators with the assumption that there is some level of correlation between features as we move forward in the sequence of one feature after the other.
- Fractional striding (for DCGAN too). If strides in CNNs reduce the dimensions of the input by the stride amount, fractional strides increase the dimensions by the stride amount (deconvolution).

Structure
- Explain GANs, generators, discriminators, losses, etc
- Explain synthetic data, context, value, and specific application for this project
- Explain CGAN
- Variational autoencoders

Possible improvements/ideas
- semantic integrity
- start with easier data types and expand from there, so maybe start with discrete or continuous values alone and then worry about combining them as an improvement
- talk about why cnn or rnn or other approaches make sense or what problems they may have on the surface



## Research Paper Architectures

The three papers at the start of the notebook present different approaches to generating tabular synthetic data, tabular data being data in a table format with rows and columns. All three use GANs for the generative aspect of their architectures. [1] and [2] come earlier than [3] and have slightly simpler architectures, though they still introduce layers of complexity that are hard to sort through. [3] comes later and looks more effective than [1] and [2], but it introduces an extra layer of complexity with a conditional GAN and more complex normalization for continuous columns.

For terminology, we will refer to the model in [1] as table-GAN, in [2] as TGAN, and in [3] as CTGAN.

Note that we are only covering new info not discussed above. Each paper has some level of review of other methods for preserving privacy and utility. The papers cover many areas related to risk, exposure, and competing methods. To keep the scope down, we will focus on the core of the approaches and architectures that each paper uses and that we can try to implement ourselves.

### [1] Data Synthesis Based on Generative Adversarial Networks

- Trains machine learning models on real and synthetic tables and shows that performance is similar -- called model compatibility or the concept that the synthetic table can replace the real table [Park et al., 2018, p. 1071]
- table-GAN can handle real tabular data that includes categorical, discrete, and continuous values and leaves other types out for now [Park et al., 2018, p. 1071]
- Consists of three ANNs: generator, discriminator, and classifier [Park et al., 2018, p. 1072]
- Classifier increases the "semantic integrity" of the synthetic data by learning the semantics in the real data, meaning the classifier learns what combinations of values in different features are legitimate so we do not end up with something like most recent course being 8th grade but having a high school graduation date [Park et al., 2018, p. 1072]
- Includes three loss functions: 1) the standard objective function for GANs with the minimax between the generator and discriminator; 2) information loss that matches the mean and standard deviation across row features, making sure that synthetic rows have the same statistical properties of real rows (based on the paper, maybe maximum-margin in hinge loss); and 3) classification loss that maintains semantic integrity, adding much more complexity (this is a step we can add in once we are ready) [Park et al., 2018, p. 1072]
- Uses DCGAN as basis for table-GAN [Park et al., 2018, p. 1073]

Overall workflow [1074]
- Convert records into square matrices with zero padding as needed
- Train table-GAN on square matrices (see details below)
- table-GAN generates synthetic square matrices that we convert into records and combine into a table
- Train models and run analysis on synthetic table
- Evaluate statistics and performance on synthetic table for evaluation purposes [Park et al., 2018, p. 1074]

table-GAN architecture [1074-5]
- DCGAN as explained above along with an additional classifier model (classifier can be an additional layer of complexity we add in after getting the remaining modeling working)
- Discriminator
    - CNN with multiple layers including batchnorm and leaky ReLU
    - Final layer is a sigmoid layer that predicts 1 for real data and 0 for synthetic data
- Generator
    - Also a CNN with multiple de-convolutional layers
    - Latent space input is a tensor with each value in the range of [-1,1]
    - De-convolutional layers convert input into a 2D matrix that matches the dimensions for the records in the synthetic table
- Classifier
    - Same architecture as the discriminator
    - Trained by ground-truth labels in the real table
    - Can train the generator if the records it produces are semantically correct, meaning are the synthetic values accurate combinations
    - Semantically incorrect synthetic records are likely to be flagged as fake by the discriminator, though the discriminator's main goal is not semantic integrity

Loss functions
- Generator uses all three losses, discriminator uses DCGAN loss, and classifier uses classification loss [1075]
- Original loss is standard GAN loss [1075]
    - Discriminator maximizes this loss while the generator minimizes it [1076]
- Information loss is the discrepancy between two statistics of synthetic and real records [1075]
    - We pull these statistics just before the sigmoid activation of the discriminator predicts the record as real or fake [1076]
    - We want this loss for both mean and standard deviation to be 0, indicating that the discriminator may not be able to distinguish between them [1076]
    - Third generator information loss lets us control deltas to set privacy levels where smaller deltas mean the synthetic data is closer to the original data. We can raise delta if we need to share synthetic data with less trustworthy recipients and want to create greater differences between the real and synthetic data. [1076]
    - These deltas are hyperparameters.
- Classification loss is the discrepancy between the label predicted by the classifier and the synthesized label [1075]

Training algorithm
- Use minibatch stochastic gradient descent (SGD) [1076]
- One issue with using SGD is that we cannot calculate the mean and standard deviation of all records for a given feature for information loss [1076]
- We use an exponentially weighted moving average to approximate the mean and standard deviation for each feature. The weight should be close to 1 for stability (paper uses 0.99) [1076-7]
- 1) train discriminator with GAN loss; 2) train classifier with classification loss for classifier; and 3) train generator with GAN loss + information loss for generator + classification loss for generator [1077]
- Once trained, we pass in latent vector z to create one synthetic record [1077]

Evaluation
- Distance to the closest record (DCR), statistical comparisons, and comparing classification or regression performance between models trained on the synthetic and the real tables [1078]

Reminder of DCGAN architecture

- Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
- Use batchnorm in both the generator and the discriminator.
- Remove fully connected hidden layers for deeper architectures.
- Use ReLU activation in generator for all layers except for the output, which uses Tanh.
- Use LeakyReLU activation in the discriminator for all layers. [Radford et al., 2016, p. 3]

### [2] Synthesizing Tabular Data using Generative Adversarial Networks

- Focus on generating tabular data with different types of data [p. 2]
- Uses long short-term memory (LSTM) with attention, so RNNs in place of CNNs in [1 -- reference, not page]
- Purposely building something different from models that create multivariate distributions for records and then sample from those distributions [p. 2]
- GANs struggle with generating discrete data [p. 2]
- An added complexity is generating multi-modal continuous data [p. 2]

Differences Between [1] and [2]
- Per [2], this paper differs from [1] in using RNNs instead of CNNs, focusing on marginal distributions more than optimizing for prediction accuracy on synthetic data by minimizing cross entropy, and explicitly learning marginal distributions for each column by minimizing Kullback-Leibler (KL) divergence [p. 2]
- KL divergence measures how different two different probability distributions are

Difficulty with Generating Synthetic Data
- Tabular data can contain different data types in different features [p. 2]
- The distributions for each feature can be "multimodal, long-tail, and several others" [2], creating an added layer of complexity for generative modeling [p. 2]

Generation
- We think of a record as having $n_c$ continuous random variables and $n_d$ discrete random variables, and one record is one sample from a joint probability distribution for each of the features [pp. 2-3]
- Each row is independent, so no sequential considerations between rows [p. 3]
- Goal is to learn a generative model that takes in each of the features and produces a synthetic table $T_{\text{synth}}$ where 1) a model learned using $T_{\text{synth}}$ achieves similar accuracy to a model trained on a test table $T_{\text{test}}$ that is set aside and compared to using the base real table T; and 2) the mutual information between two arbitrary variables in the synthetic or the real table is similar [p. 3]
- See how similar or different this is from the proposed goals in [1 -- paper]

Reversible Data Transformation
- ANNs can effectively generate continuous distributions centered over [-1,1] using tanh and low-cardinality multinomial distributions using softmax [p. 3]
- So we convert continuous variables to scalars in range [-1,1] and multinomial distributions and also convert discrete variables into multinomial distributions [p. 3]

Mode-Specific Normalization
- Numeric variables can follow a multimodal distribution [p. 3]
- We use Gaussian kernel density estimation (KDE) to estimate the number of modes for continuous variables [p. 3]
- When continuous variables have more than one mode, we run into problems when we use tanh to generate these in the range [-1,1], causing problems with backpropagation if any of the modes are near -1 or 1 or at least not near the center or 0 for tanh [p. 3]
- We use Gaussian mixture models (GMM) to cluster each continuous random variable to determine the number of modes, and then we sample from the GMM. The paper uses m=5 clusters for all GMMs since GMMs will weight each m and will provide low weights for ms in scenarios where there are less than 5 modes for a variable [p. 3]

Smoothing
- A core problem for generating categorical variables is that discrete variables are not differentiable, so they fail when we go through backpropagation. We get around this by one-hot encoding categorical variables and then using softmax along with adding noise to the binary variables to create differentiable units [p. 3]
- The paper uses $\gamma = 0.2$ for the random noise added to the one-hot encoded binary variables [p. 3]

Generation Part Deux
- LSTM for the generator
- Multi-layer perceptron (MLP) for the discriminator [p. 4]
- Two steps for generating numerical variables: 1) generate the scalar value $v_i$; and 2) the cluster vector $u_i$ that weights which mode/cluster the value comes from [p. 4]
- One step for generating categorical variables: probability distribution over all the possible values [p. 4]
- Return and explain the details of building the LSTM RNN if we test out implementing this one. There are a number of pieces explained in the paper. [p. 4]
- The discriminator is pretty straightforward but does require looking up another paper for the diversity vector [p. 4]

Loss
- The generator uses standard GAN loss but adds the KL divergence of the discrete variables and the KL divergence of the cluster vectors of continuous variables [p. 4]
- The discriminator uses conventional cross-entropy loss [p. 4]

Evaluation
- Machine learning efficacy and pairwise mutual information as described below

### [3] Modeling Tabular Data using Conditional GAN

- GANs have issues with generating synthetic data, including modeling both continuous and discrete random variables, multi-modal non-Gaussian continuous columns, and severe imbalance of categorical columns [p. 1]
- The proposed CTGAN introduces several new techniques: 1) mode specific normalization; 2) architectural changes; 3) and addressing imbalances with a conditional generator and training-by-sample [p. 2]
- Introduces tabular variational autoencoder (TVAE) that uses a variational autoencoder to produce tabular data, though the authors found that CTGAN outperforms TVAE [p. 2]
- Introduces a synthetic data benchmarking system that is open source and extensible [p. 2]. We may want to look at using this to evaluate different models that we build.
- Similar setup to other papers where we set up T_train, T_test, and T_syn [p. 2]
- We evaluate the generator by looking at 1) if T_syn follows the same joint probability distribution as T_train; and 2) if we train a classifier or regressor on T_test to predict one column based on the other columns, do we see similar performance using a model trained on T_syn as one trained on T_train [p. 2]

Issues/Difficulties
- Tabular data has discrete and continuous columns, so we need to use softmax and tanh in combination [p. 3]
- Continuous columns are often non-Gaussian, so applying a tanh activation to scrunch the output into [-1,1] with a minmax function leads to vanishing gradients [p. 3]
- Continuous columns often are multi-modal, and vanilla GANs struggle with modeling multiple modes on 2D data (there is a separate paper that explores GANs modeling multi-modal continuous data if needed) [p. 3]
- Real discrete data is converted to one-hot encoded vectors while synthetic generated discrete data is a probability distribution over all factor levels with softmax, and a trivial discriminator can tell the difference between these two [p. 3]
- Highly imbalanced discrete columns result in mode collapse and a lack of representation for minority classes [p. 3]

Mode-Specific Normalization
- A traditional approach for normalize continuous variables is to use a minmax to put values into the range of [-1,1], but this causes problems when these variables have more than one mode [p. 3]
- For each continuous column, we use variational GMMs to estimate the number of modes and fit a Gaussian mixture, calculate the probability of each value coming from each mode, and then pick one mode to normalize the value [p. 3-4]
- One row ends up becoming the concatenation of each of the columns after mode-specific normalization and one-hot encoding for continuous and discrete columns respectively [p. 4]

Condition Generator and Training-by-Sample
- The core issue we address here is class imbalance. The minority class shows up few enough times that the generator may not reliably learn to generate those values. If we decide to resample, then the generator learns a resampled distribution, not the true distribution. [p. 4]
- The goal is to resample such that all classes are resampled evenly -- not uniformly -- and that we can recover the real data distribution -- not resampled -- during test [p. 4]
- We can look at conditional distributions predicated on each class value for a given column, so the generator is the conditional distribution of rows given a particular value in a particular column. The paper calls this a conditional generator and a GAN built on it a conditional GAN. [p. 4]
- Integrating a conditional GAN into the architecture creates a few issues: 1) need to create a representation for the condition and an input for it; 2) the generated rows must retain the condition: and 3) the conditional generator must learn the real conditional distribution, so P_{generator}(row|class) = P_{real}(row|class). We can recreate the original distribution by multiplying P_{generator}(row|class) P(class in real distribution) [p. 4]
- The paper proposes a solution that uses 1) the conditional vector; 2) the generator loss; and 3) the training-by-sampling method [p. 4]

Conditional Vector
- We one-hot encode discrete columns. For each one-hot encoded discrete column, we can put a mask of 1 when the value for that column matches the right one-hot encoded column. We repeat this for each discrete column and then concatenate each of the mask vectors together to create the conditional vector (see example in paper). [p. 5]

Generator Loss
- One issue is that the generator can create conditional vectors with incorrect masks where we put a 1 for a mask for a discrete column that should have all 0s. We can add a cross entropy loss between m and d averaged over all the instances of the batch. [p. 4]

Training-by-Sampling
- Create a new conditional vector with 0 masks for all classes for all columns. Randomly select a discrete column, and then calculate the probability mass function for the classes in that column such that the probability mass of each value is the logarithm of its frequency in that column. Sample from the PMF to get a class and set its associated mask to 1. Combine the mask vectors into a conditional vector as described earlier. [p. 5]
- The discriminator calculates the distance between a generated conditional vector and a training-by-sampling conditional vector [p. 5]

Network Structure
- We use MLPs with two fully connected hidden layers in both the generator and discriminator since rows in a tabular dataset do not have local structure. The MLPs capture all possible combinations of columns. [p. 6]
- The generator MLP uses batchnorm and ReLU for the hidden layers. After the two hidden layers, the synthetic row representation is generated by a mix of tanh for scalar values and by gumbel softmax for both the mode indicators and discrete values. [p. 6]
- The discriminator uses leaky ReLU and dropout on each hidden layer [p. 6]
- The conditional generator architecture is laid out in the paper with the layers separated out [p. 6]
- The discriminator uses PacGAN to help with mode collapse (another paper to look up if needed) [p. 6]
- Train the model with WGAN (another paper to look up if needed) [p. 6]
- Adam optimizer [p. 6]

Tabular Variational Autoencoder (TVAE)
- Add details if needed, though this may be one too many models to test out, especially since the paper points out that CTGAN outperforms TVAE

Benchmarking Tool
- The authors of this paper created [SDGym](https://github.com/sdv-dev/SDGym) and open sourced it. We could look at using this to evaluate models for this project. The paper points out how other papers differ in how they benchmark synthetic generation models, so standardizing on a benchmarking tool could provide better comparative insight between models.
