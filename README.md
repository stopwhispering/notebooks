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



