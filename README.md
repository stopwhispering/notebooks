## Neural Networks

### custom_torch_cbow_embeddings_from_150novels.ipynb
Subject: Building a Word2vec-like CBOW Model to create embeddings for the dataset's words representing their near closeness to each other (in 100 embedding dimensions).

Data: txtlab_Novel150_English (150 English novels from the 19th century)

Procedure:
- Tokenizing with nltk.tokenize.word_tokenize and nltk.corpus.stopwords
- Creating contexts and targets from five words each: (01 34) with (2) as target
- Tensorizing contexts and targets
- Creating a vocabulary with torchtext.vocab.build_vocab_from_iterator
- Creating a custom torch.utils.data.Dataset for a torch.utils.data.DataLoader
- Word2vec-like CBOW model with torch.nn.module, torch.nn.Embedding, torch.nn.Linear, torch.nn.ReLU,  and torch.nn.LogSoftmax
- Training with torch.nn.NLLLoss, torch.optim.SGD, and torch.optim.lr_scheduler.StepLR
- Evaluation by finding some nearest words and playing with word vectors
- Disappointing results (probably much more data required)

Others:
- CUDA support
- working on Colab with Google Drive for saving/loading interim stages


### custom_torch_cbow_embeddings_from_wikitext2.ipynb
Subject: Building a Word2vec-like CBOW Model to create embeddings for the dataset's words representing their near closeness to each other (in 100 embedding dimensions).

Data: WikiText-2 via torchtext 

Procedure:
- Tokenizing with torchtext.data.utils.get_tokenizer and nltk.corpus.stopwords
- Creating contexts and targets from five words each: (01 34) with (2) as target
- Tensorizing contexts and targets
- Creating a vocabulary with torchtext.vocab.vocab
- Creating a custom torch.utils.data.Dataset for a torch.utils.data.DataLoader
- Word2vec-like CBOW model with torch.nn.module, torch.nn.Embedding, torch.nn.Linear, torch.nn.ReLU,  and torch.nn.LogSoftmax
- Training with torch.nn.NLLLoss, torch.optim.SGD, and torch.optim.lr_scheduler.StepLR
- Evaluation by finding some nearest words and playing with word vectors
- Disappointing results (probably much more data required)

Others:
- CUDA support
- working on Colab with Google Drive for saving/loading interim stages


### custom_torch_cbow_embeddings_from_wikitext3.ipynb
Subject: Building a Word2vec-like CBOW Model to create embeddings for the dataset's words representing their near closeness to each other (in 300 embedding dimensions). Preprocessing fails on Colab due to data size, therefore we process data as <b>iterators</b>, hot having the whole dataset in memory at any time.

Data: Wikitext3 from torch as DataPipe (beta)

Procedure:

- Creating a vocabulary with torchtext.vocab.build_vocab_from_iterator
- Tokenizing with torchtext.data.utils.get_tokenizer and nltk.corpus.stopwords
- Creating contexts and targets from five words each: (01 34) with (2) as target
- Tensorizing contexts and targets
- Creating a custom torch.utils.data.Dataset for a torch.utils.data.DataLoader
- Word2vec-like CBOW model with torch.nn.module, torch.nn.Embedding, torch.nn.Linear, torch.nn.ReLU, and torch.nn.LogSoftmax
- Training with torch.nn.NLLLoss, torch.optim.SGD, and torch.optim.lr_scheduler.StepLR
- Evaluation by finding some nearest words and playing with word vectors
- Disappointing results (probably much more data required)

Others:
- CUDA support
- working on Colab with Google Drive for saving/loading interim stages


### simple_torch_lstm_word_generator_from_wikitext2.ipynb
Subject: Building a demo word prediction model.

Data: WikiText-2 via torchtext 

Procedure:
- Creating a vocabulary as simple set of distinct words
- Tokenizing simply via '...'.split()
- Creating a tensor from the indices of the flat word list
  Train is tensor of indices of 35 subsequent words, Target is the same but with one word further. (e.g. 0, 1, ..., 34 -> 1, 2, ..., 35)
- RNN Model with Long short-term memory (LSTM) and embeddings layer. Using torch.nn.Dropout, torch.nn.Embedding, torch.nn.LSTM, torch.nn.Linear, and torch.nn.functional.log_softmax.
- Training with torch.nn.NLLLoss, no optimizer (?)
- Evaluation by generating some words
- Disappointing results (to be expected with that small dataset)

Others:
- CUDA support
- working on Colab with Google Drive for saving/loading interim stages


### simple_torch_ngram_embeddings_from_example_text.ipynb
Subject: Building a model to create embeddings for the dataset's words representing their near closeness to each other. Training with two subsequent words as train and the subsequent third word as target (N-GRAMS).

Data: Static string (just as an example)

Procedure:
- Tokenizing simply via '...'.split()
- Creating contexts and targets from three words each: (0, 1) with (2) as target
- Creating a vocabulary as simple set of distinct words
- Tensorizing contexts and targets
- Word2vec-like CBOW model with torch.nn.module, torch.nn.Embedding, torch.nn.Linear,  and torch.nn.functional.log_softmax
- Training with torch.nn.NLLLoss, and torch.optim.SGD
- Evaluation by finding some nearest words and playing with word vectors
- Disappointing results (to be expected with that simple input data)


### torch_vanilla_rnn_for_family_name_language_prediction.ipynb
Subject: Predicting the Language Origin of family names with a custom vanilla RSTM Model (no LSTM).

Data: Family Names in 18 Languages (from pytorch tutorial)

Procedure:

- Using string.ascii_letters list as a simple vocabulary
- Tokenizing characters in the family names
- Tensorizing train and target, with targets being the 18 languages
- Creating a torch.utils.data.TensorDataset and torch.utils.data.DataLoader
- Creating torch.nn.module model with torch.nn.Linear, torch.nn.LogSoftmax, and torch.tanh
- Training with torch.nn.CrossEntropyLoss, torch.optim.Adam, and hidden state tensors for storing the state in between a sequence
- Visualization of losses with pyplot
- Evaluation by classifying arbitrary input family names
- Quite encouraging results

Others:
- CUDA support
- working on Colab with Google Drive for loading data


### torch_lstm_rnn_for_family_name_language_prediction_padding.ipynbSubject: Predicting the Language Origin of family names with a custom vanilla RSTM Model (no LSTM).
Subject: Predicting the Language Origin of family names with an LSTM Model.

Data: Family Names in 18 Languages (from pytorch tutorial)

Procedure:

- Using string.ascii_letters list as a simple vocabulary
- Tokenizing characters in the family names
- Tensorizing train and target, with targets being the 18 languages
- Creating a torch.utils.data.TensorDataset and torch.utils.data.DataLoader
- We're training with batches. As all names need to have the same length in tensors, we apply a padding approach: The tensors have the size of the overall longest name (19 characters). Smaller names are filled up to that size with zeros. Cf. other Notebook where a different approach (with better results) is applied.
- Creating torch.nn.module model with torch.nn.Linear, and torch.nn.LogSoftmax
- Training with torch.nn.CrossEntropyLoss, torch.optim.Adam, and hidden state tensors for storing the state in between a sequence
- Visualization of losses with pyplot
- Evaluation by classifying arbitrary input family names
- Quite encouraging results

Others:
- CUDA support
- working on Colab with Google Drive for loading data


### torch_lstm_rnn_for_family_name_language_prediction_var_batch_length.ipynb
Subject: Predicting the Language Origin of family names with an LSTM Model.

Data: Family Names in 18 Languages (from pytorch tutorial)

Procedure:

- Using string.ascii_letters list as a simple vocabulary
- Tokenizing characters in the family names
- Tensorizing train and target, with targets being the 18 languages
- Creating a torch.utils.data.TensorDataset and torch.utils.data.DataLoader
- We're training with batches. As all names need to have the same length in tensors, we apply a <b>variable batch length approach</b>: The training data is grouped into batches of the same name lenghts. Cf. other Notebook where a padding approach (with worse results) is applied.
- Creating torch.nn.module model with torch.nn.Linear, and torch.nn.LogSoftmax
- Training with torch.nn.CrossEntropyLoss, torch.optim.Adam, and hidden state tensors for storing the state in between a sequence
- Visualization of losses with pyplot
- Evaluation by classifying arbitrary input family names
- Quite encouraging results

Others:
- CUDA support
- working on Colab with Google Drive for loading data


### create_word2vec_embedding_with_gensim_from_150novels.ipynb
Subject: Training word embeddings with gensim Word2vec (CBOW algorithm) representing the words' closeness to each other.

Data: Unlabeled "txtlab_Novel150_English" dataset with 150 English novels written over the long nineteenth century

Procedure:
- Tokenizing to sentences with nltk's sentence tokenizer nltk.data.load('tokenizers/punkt/english.pickle') 
- Tokenizing words with nltk.tokenize.word_tokenize
- Training a gensim.models.word2vec.Word2Vec model
- Evaluation with gensim's model.wv.doesnt_match(), model.wv.most_similar(), model.wv.n_similarity(), model.wv.most_similar_cosmul() including the canonical King - Man + Woman -> Queen test (passed)
- Wordclouds for male and female-associated words
- Reduce the dimensionality from 200 to 2 via Principal Component Analysis (PCA) and display the two remaining dimensions of the most common words in a scatterplot to show near distance.
- Good results (not reproduced with custom Torch models, cf. other notebooks)

Others:
- working on Colab with Google Drive for loading data


### torch_cnn_for_mnist_classification.ipynb
Subject: Building a CNN Classificator with PyTorch to classify hand-written digits.

Data: MNIST (handwritten digits) via torchvision

Procedure:
- Previewing images from dataset with pyplot's imshow()
- Neural network with torch.nn.module, torch.nn.Sequential, torch.nn.Conv2d, torch.nn.ReLU, torch.nn.MaxPool2d, torch.nn.Dropout, torch.nn.Linear, and torch.nn.init.xavier_uniform_
- Visualizing model with torchviz' make_dot()
- Training with nn.CrossEntropyLoss and torch.optim.Adam optimizer
- Visualization Loss and Accuracy with pyplot
- Visualization of the CNN Layers with pyplot's imshow()
- Good results

Others:
- CUDA support


### torch_simple_linear_nn_for_iris_classification.ipynb
Subject: Simple, linear neural network with Torch for IRIS classification.

Data: Canonical Iris dataset (via sklearn.datasets.load_iris())

Procedure:
- visualize data distribution
- split into train/test with torch.utils.data.random_split
- tensorize and create from torch.utils.data.DataLoader
- scale with sklearn.preprocessing.StandardScaler
- neural network with torch.nn.Module, consisting of input and two hidden torch.nn.Linear layers, with torch.nn.functional.relu inbetween and torch.nn.functional.softmax at the end
- visualize model with torchviz' make_dot()
- training with torch.nn.CrossEntropyLoss and torch.optim.Adam
- visualize loss and accuracy
- display a ROC curve
- (expected) good results

Others:
- CUDA support


### torch_generate_digits_with_linear_NN_GAN_from_mnist.ipynb
Subject: Building a GAN (Generative Adversarial Network) with PyTorch to generate hand-written digits from noise, trained from MNIS dataset.

Data: MNIST (handwritten digits) via torchvision

Procedure:
- Previewing images from dataset with pyplot's imshow()
- Generator network with torch.nn.module, torch.nn.Sequential, torch.nn.Linear, torch.nn.Sigmoid, torch.nn.Linear, torch.nn.BatchNorm1d, torch.nn.LeakyReLU, and torch.nn.Dropout
- Discriminator with torch.nn.module, torch.nn.Sequential, torch.nn.Linear, and torch.nn.LeakyReLU,
- Visualizing generator and discriminator with torchviz' make_dot()
- Training with nn.BCEWithLogitsLoss as loss and two torch.optim.Adam optimizers (one for both generator and discriminator)
- Previewing generated digits with imshow() after each n training steps
- Visualization Loss for generator and discriminator with pyplot
- Display development of generated images during training as image show with matplotlib.animation.ArtistAnimation
- Interesting results

Others:
- CUDA support


### torch_generate_digits_with_CNN_GAN_from_mnist.ipynb
Subject: Building a GAN (Generative Adversarial Network) with PyTorch to generate hand-written digits from noise, trained from MNIS dataset. Use <b>CNN Architecture</b> (cf. other notebook).

Data: MNIST (handwritten digits) via torchvision

Procedure:
- Previewing images from dataset with pyplot's imshow()
- Generator network with torch.nn.module, torch.nn.Sequential, torch.nn.BatchNorm2d, torch.nn.Upsample, torch.nn.Conv2d, torch.nn.LeakyReLU, and torch.nn.Tanh
- Discriminator with torch.nn.module, torch.nn.Sequential, torch.nn.Conv2d, and torch.nn.LeakyReLU, torch.nn.Dropout2d, torch.nn.Linear, and torch.nn.Sigmoid
- Visualizing generator and discriminator with torchviz' make_dot()
- Training with nn.BCEWithLogitsLoss as loss and two torch.optim.Adam optimizers (one for both generator and discriminator)
- Previewing generated digits with imshow() after each n training steps
- Visualization Loss for generator and discriminator with pyplot
- Display development of generated images during training as image show with matplotlib.animation.ArtistAnimation
- Interesting results

Others:
- CUDA support


pretrained_word2vec_google_news_gensim_model.ipynb
Subject: Download and work with pretrained Work2vec embeddings.

Data: google-news model, trained with Google News dataset (about 100 billion words). The model contains 300-dimensional vectors for 3 million words and phrases. 

Procedure:
- Print list of pretrained models from Gensim
- Download model trained with Google-News dataset (to be executed in console)
- Save that model locally
- Load that model from local file, preferably with limited vocab
- Load the model's vocab in Pytorch as torchtext.vocab.Vocab
- Load the model's word vectors into a torch.nn.Embedding layer for usage in torch model

Others:
- Support for Colab for saving/loading from Google Drive







