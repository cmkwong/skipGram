import numpy as np
import torch
import random
import os

class Trainer:
    def __init__(self, data_processor, model, optimizer, criterion, noise_dist, checkpoint_path):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.noise_dist = noise_dist
        self.data_processor = data_processor
        self.checkpoint_path = checkpoint_path
        self.steps = 0

    def cosine_similarity(self, embedding, valid_size=16, valid_window=100, device='cpu'):

        embed_vectors = embedding.weight

        # magnitude of embedding vectors, |b|
        magnitudes = embed_vectors.pow(2).sum(dim=1).sqrt().unsqueeze(0)

        # pick N words from our ranges (0,window) and (1000,1000+window). lower id implies more frequent
        valid_examples = np.array(random.sample(range(valid_window), valid_size // 2))
        valid_examples = np.append(valid_examples,
                                   random.sample(range(1000, 1000 + valid_window), valid_size // 2))
        valid_examples = torch.tensor(valid_examples, dtype=torch.long).to(device)

        valid_vectors = embedding(valid_examples)
        similarities = torch.mm(valid_vectors, embed_vectors.t()) / magnitudes

        return valid_examples, similarities

    def update_tensorboard(self, loss, step):
        pass

    def train(self, sentcs, epochs, batch_size, window_size, print_every, checkpoint_step=150000, device="cuda"):

        self.model = self.model.to(device)
        for e in range(epochs):
            # get our input, target batches
            for input_words, target_words in self.data_processor.get_batches(sentcs, batch_size=batch_size, window_size=window_size, BUFFER_SIZE=100000):
                self.steps += 1
                inputs, targets = torch.tensor(input_words, dtype=torch.long).to(device), \
                                  torch.tensor(target_words, dtype=torch.long).to(device)

                # input, output, and noise vectors
                input_vectors = self.model.forward_input(inputs)
                output_vectors = self.model.forward_output(targets)
                noise_vectors = self.model.forward_noise(inputs.shape[0], 5)

                # negative sampling
                # loss
                loss = self.criterion(input_vectors, output_vectors, noise_vectors)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # loss stats
                if self.steps % print_every == 0:
                    print("Epoch: {}/{} - step: {}".format(e + 1, epochs, self.steps))
                    print("Loss: ", loss.item())  # avg batch loss at this point in training
                    valid_examples, valid_similarities = self.cosine_similarity(self.model.in_embed, device=device)
                    _, closest_idxs = valid_similarities.topk(6)

                    valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
                    for ii, valid_idx in enumerate(valid_examples):
                        closest_words = [self.data_processor.int2word[idx.item()] for idx in closest_idxs[ii]][1:]
                        print(self.data_processor.int2word[valid_idx.item()] + " | " + ', '.join(closest_words))
                    print("...\n")

                if self.steps % checkpoint_step == 0:
                    checkpoint = {
                        "state_dict": self.model.state_dict()
                    }
                    with open(os.path.join(self.checkpoint_path, "checkpoint-%d.data" % self.steps), "wb") as f:
                        torch.save(checkpoint, f)


