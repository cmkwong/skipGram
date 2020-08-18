from lib import data, models, validation, common
import torch
from torch import optim
import numpy as np
import os
import re


DATA_PATH = "../data"
target_file_path = "en-zh/UNv1.0.en-zh.zh"

MAIN_PATH = "../docs/1"
NET_SAVE_PATH = MAIN_PATH + '/checkpoint'
NET_FILE = "checkpoint-1500000.data"
BATCH_SIZE = 64
lr = 0.00001
PREPROCESS_DONE = True
LOAD_NET = True
CHECKPOINT_STEP = 500000
PRINT_EVERY = 5000

data_processor = data.Data_Processor_Sentc(rare_word_threshold=5, sampling_threshold=1e-5)

if PREPROCESS_DONE:
    print("Reading token count file...")
    token_count = data_processor.read_token_count_file(DATA_PATH, "token_count.csv")
    print("Successful!")
    print("Reading sentcs file...")
    sentcs = data_processor.read_sentcs_file(DATA_PATH, "sentcs.csv")
    print("Successful!")
    print("Reading word-int pair file...")
    data_processor.read_word_int_pair_file(DATA_PATH, "wordIntPair.csv")
    print("Successful Update!")
else:
    sentcs, token_count = data_processor.preprocess_text(main_path=DATA_PATH, target_file_path=target_file_path,
                                                         filter_rare_words=False, sampling_words=False)

# define model
skip_gram_model = models.SkipGramNeg(n_vocab=len(token_count), n_embed=1024)
if LOAD_NET:
    print("Loading net params...")
    with open(os.path.join(NET_SAVE_PATH, NET_FILE), "rb") as f:
        checkpoint = torch.load(f)
    skip_gram_model.load_state_dict(checkpoint['state_dict'])
    print("Successful!")

# define the sampling hist
word_freqs = np.array(sorted(token_count.values(), reverse=True))
unigram_dist = word_freqs/word_freqs.sum()
noise_dist = torch.from_numpy(unigram_dist**(0.75)/np.sum(unigram_dist**(0.75)))

# define criterion
criterion = validation.NegativeSamplingLoss()

# define optimizer
optimizer = optim.Adam(skip_gram_model.parameters(), lr=lr)

# define trainer
trainer = common.Trainer(data_processor, skip_gram_model, optimizer, criterion, noise_dist, checkpoint_path=NET_SAVE_PATH)
# update the steps if net is loaded
if LOAD_NET:
    trainer.steps = int(re.match('checkpoint-([\d]*)', NET_FILE).group(1))

# training begin
trainer.train(sentcs, epochs=5, batch_size=512, window_size=3, print_every=PRINT_EVERY, checkpoint_step=CHECKPOINT_STEP, device="cuda")

