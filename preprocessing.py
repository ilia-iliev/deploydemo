import re
from collections import defaultdict
import pickle
import torch

def remove_multiple_whitespace(s):
    return re.sub(' +', ' ', s.lower())

class Tokenizer():
    # Implementation of Byte Pair Encoding tokenizer
    def __init__(self, num_tokens=350, min_occurances=10) -> None:
        self.token_ids = None
        self.n_tokens = num_tokens
        self.min_occurances = min_occurances
        self.PAD = num_tokens-1
        self.SOS = num_tokens-2
        self.EOS = num_tokens-3
        self.regex_pattern, self.token_mapping, self.regex_pattern = None, None, None
    
    def fit(self, data):
        data = [remove_multiple_whitespace(datapoint) for datapoint in data]
        one_big_string = ''.join(data)
        unique_chars = ''.join(set(one_big_string))
        
        token_ids = [char for char in unique_chars if one_big_string.count(char)>=self.min_occurances]
        self.update_regex(token_ids)
        couples = defaultdict(int)
        while len(self.token_mapping)+3 < self.n_tokens:    # add threehardcoded special characters
            for datapoint in data:
                tokenized = self._tokenize(datapoint)
                for i in range(len(tokenized)-2):
                    # how many times each pair of embeddings is encountered
                    couples[tokenized[i], tokenized[i+1]] += 1
            max_couple = max(couples, key=couples.get)
            token_ids.append(token_ids[max_couple[0]]+token_ids[max_couple[1]])
            self.update_regex(token_ids)
            couples = defaultdict(int)
            
    def update_regex(self, token_ids):
        self.token_mapping = {value: index for index, value in enumerate(token_ids)}
        reg = ('|'.join(re.escape(token) for token in sorted(token_ids, key=len, reverse=True)))
        self.regex_pattern = re.compile(f"{reg}")


    def _tokenize(self, data):
        regex_list = re.findall(self.regex_pattern, data)
        return [self.token_mapping[id] for id in regex_list]
    
    def tokenize(self, data, padding_size=0):
        tokens = self._tokenize(data)
        return [self.SOS, *tokens, self.EOS, *[self.PAD] * (padding_size-(2+len(tokens)))]
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer


class TokenizingDataset(torch.utils.data.Dataset):
    def __init__(self, ds, tokenizer=None, seq_len=None):
        if not tokenizer:
            tokenizer = Tokenizer()
            tokenizer.fit(ds['text'])
        self.tokenizer = tokenizer

        self.len = len(ds['text'])

        lengths = [len(t) for t in [self.tokenizer.tokenize(text) for text in ds['text']]]
        if not seq_len:
            seq_len = max(lengths)
        
        self.tokens = torch.IntTensor([self.tokenizer.tokenize(text, seq_len) for text in ds['text']])
        mask = [[1 for i in range(seq_len+1)] for j in range(self.len)]
        for i, length in enumerate(lengths):
            mask[i][0:length] = [0]*(length-1)
        self.mask=torch.BoolTensor(mask)
        self.labels = torch.tensor(ds['label'])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.tokens[idx], self.mask[idx], self.labels[idx]

