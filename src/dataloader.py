import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

# dataset implementation borrowed heavily from GPTDatasetV1 from Rachka's book
class model_data(Dataset):
    def __init__(self, text, tokenizer = 'gpt2', max_length = 256, stride = 128, batch_size = 4, shuffle = True, drop_last = True, num_workers = 0):
        self.tokenizer = tiktoken.get_encoding(tokenizer)
        self.input_ids = []
        self.target_ids = []

        # tokenize the text
        token_ids = self.tokenizer.encode(text, allowed_special = {"<|endoftext|>"})

        # Use a sliding window to chunk the text into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i + max_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1: i + max_length + 1]))
        
        self.data = self.create_dataloader(batch_size = batch_size, shuffle = shuffle, drop_last = drop_last, num_workers = num_workers)
        
    def create_dataloader(self, batch_size = 4, shuffle = True, drop_last = True, num_workers = 0):
        return DataLoader(self, batch_size = batch_size, shuffle = shuffle, drop_last = drop_last, num_workers = num_workers)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
