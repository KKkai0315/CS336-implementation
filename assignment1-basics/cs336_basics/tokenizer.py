import os
import regex as re
import pathlib
import pickle
from collections import Counter, defaultdict
from typing import Iterable, Union

class BPETokenizer:
    def __init__(self, 
                 vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] = []
                 ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.bytes_to_id = {v: k for k, v in vocab.items()}
        self.bpe_rank = dict(zip(merges, range(len(merges))))
        for token in special_tokens:
            token = token.encode("utf-8")
            if token not in self.bytes_to_id:
                new_id = max(self.bytes_to_id.values())+1
                self.bytes_to_id[token] = new_id
                self.vocab[new_id] = token

    @classmethod
    def from_files(cls,
                   vocab_filepath: str | os.PathLike,
                   merges_filepath: str | os.PathLike,
                   specials_tokens: list[str] | None = None
                   ):
        pass

    @classmethod
    def from_pickle(cls,
                    vocab_filepath: str | os.PathLike,
                    merges_filepath: str | os.PathLike,
                    special_tokens: list[str] | None = None
                    ):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        sorted_special_tokens = sorted(self.special_tokens, key=lambda x: len(x), reverse=True)
        if not sorted_special_tokens:
            return self.tokenize(text)
        pattern = '|'.join(map(re.escape, sorted_special_tokens))
        pattern = f"({pattern})" if pattern else None
        
        tokens = []
        if pattern:
            parts = re.split(f"{pattern}", text)
        else:
            parts = [text]
        
        for part in parts:
            if part in self.special_tokens:
                tokens.append(self.bytes_to_id[part.encode("utf-8")])
            else:
                tokens.extend(self.tokenize(part))
        return tokens

    def encode_iterable(self,
                        iterable: Iterable[str],
                        )-> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        full_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return full_bytes.decode('utf-8', errors='replace')
          
    def tokenize(self, text: str) -> list[int]:
        def word2tuple(word):
            if isinstance(word, str):
                word = word.encode('utf-8')
            word = list(word)
            word = [bytes([b]) for b in word]
            return tuple(word)
    
        pre_tokens = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        for m in re.finditer(PAT, text):
            word = m.group(0)
            pre_tokens.append(word.encode('utf-8'))
        
        token_ids = []
        for token in pre_tokens:
            token_tuple = word2tuple(token)
            merged = self.merge(token_tuple)
            token_ids.extend(self.bytes_to_id[b] for b in merged)
        return token_ids
    
    def merge(self, byte_tuple: tuple[bytes]) -> list[bytes]:
        def get_pairs(word: list[bytes]) -> set[tuple[bytes, bytes]]:
            pairs = set()
            for i in range(len(word) - 1):
                pairs.add((word[i], word[i + 1]))
            return pairs
        
        pairs = get_pairs(list(byte_tuple))
        word = list(byte_tuple)
        if not pairs:
            return list(byte_tuple)
        
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_rank.get(pair, float('inf')))
            if bigram not in self.bpe_rank:
                break
            new_word = []
            first, second = bigram
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j
                if word[i] == first and i + 1 < len(word) and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                return list(word)
            pairs = get_pairs(word)
        return list(word)