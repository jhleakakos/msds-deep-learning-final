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

Structure
- Explain GANs, generators, discriminators, losses, etc
- Explain synthetic data, context, value, and specific application for this project
- Explain CGAN

References
- https://www2.stat.duke.edu/~jerry/Papers/jos03.pdf 
- https://www2.stat.duke.edu/~jerry/Papers/sm04.pdf
- https://arxiv.org/abs/2303.01230v3
- https://dl.acm.org/doi/10.1145/3636424
- https://arxiv.org/abs/1907.00503
- https://arxiv.org/abs/1609.05473
- https://arxiv.org/abs/1611.04051
- https://arxiv.org/abs/1810.06640
- https://arxiv.org/abs/2403.04190v1
- https://arxiv.org/abs/1811.11264
- https://medium.com/@aldolamberti/synthetic-data-101-synthetic-data-vs-real-dummy-data-237d790433a9
- https://machinelearningmastery.com/mostly-generate-synethetic-data-machine-learning-why/
- https://towardsdatascience.com/generative-ai-synthetic-data-generation-with-gans-using-pytorch-2e4dde8a17dd
- https://becominghuman.ai/generative-adversarial-networks-for-text-generation-part-1-2b886c8cab10
- https://becominghuman.ai/generative-adversarial-networks-for-text-generation-part-3-non-rl-methods-70d1be02350b
- https://towardsdatascience.com/how-to-generate-tabular-data-using-ctgans-9386e45836a6
- https://medium.com/analytics-vidhya/a-step-by-step-guide-to-generate-tabular-synthetic-dataset-with-gans-d55fc373c8db
- https://github.com/sdv-dev/TGAN
- https://github.com/sdv-dev/CTGAN
- https://www.youtube.com/watch?v=yujdA46HKwA (GANs for Tabular Synthetic Data Generation)
- https://www.youtube.com/watch?v=Ei0klF38CNs (Synthetic data generation with CTGAN)
- https://www.youtube.com/watch?v=ROLugVqjf00 (Generation of Synthetic Financial Time Series with GANs - Casper Hogenboom)
- https://www.youtube.com/watch?v=HIusawrGBN4 (What is Synthetic Data? No, It's Not "Fake" Data)
- https://www.youtube.com/watch?v=FLTWjkx0kWE (Generate Synthetic Tabular Data with GANs)
- https://www.youtube.com/watch?v=zC3_kM9Qwo0 (QuantUniversity Summer School 2020 | Generating Synthetic Data with (GANs))

Datasets
- https://www.kaggle.com/datasets/alejopaullier/pii-external-dataset
- https://www.kaggle.com/datasets/pjmathematician/pii-detection-dataset-gpt
- https://www.kaggle.com/datasets/newtonbaba12345/pii-detection-gemini-created-dataset

Modeling Tabular Data Using Conditional GAN
- Modeling the probability distribution of rows in tabular data
- Generate synthetic data from that row-based probability distribution
- Tabular data has a mix of discrete and continuous columns
- Designed CTGAN with a conditional generator to address challenges