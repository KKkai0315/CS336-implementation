import os
import regex as re
import pathlib
import pickle
from jaxtyping import Int, Float
from collections import Counter,defaultdict
from tqdm import tqdm


def train_tokenizer_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
)-> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = dict[int, bytes]()
    merges = list[tuple[bytes, bytes]]()
    for i in range(256):
        vocab[i] = bytes([i])
    next_token_id = 256
    # Add special tokens to the vocabulary
    for special_token in special_tokens:
        if special_token not in vocab.values():
            vocab[next_token_id] = special_token.encode('utf-8')
            next_token_id += 1
    
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Preprocess text
    pre_token_cnt = preprocess_test(text,special_tokens)

    total_merges = vocab_size - len(vocab)
    pbar = tqdm(total=total_merges, desc="Training BPE tokenizer")

    while len(vocab) < vocab_size:
        pair_counts = defaultdict(int)
        for word, count in pre_token_cnt.items():
            if len(word) < 2:
                continue
            for i in range(len(word)-1):
                pair = (word[i], word[i+1])
                pair_counts[pair] += count

        if not pair_counts:
            break

        max_count = max(pair_counts.values())
        candidates = [pair for pair, count in pair_counts.items() if count == max_count]
        max_pair = max(candidates)

        a,b = max_pair
        new_token = a+b
        vocab[next_token_id] = new_token
        next_token_id += 1
        merges.append((a,b))
        
        pbar.update(1)
        pbar.set_postfix({"Current vocab size": len(vocab), "Max pair count": max_count})

        new_pre_token_cnt = defaultdict(int)
        for word, count in pre_token_cnt.items():
            new_word = []
            skip = False
            for i in range(len(word)):
                if skip:
                    skip = False
                    continue
                if i < len(word) - 1 and (word[i], word[i+1]) == max_pair:
                    new_word.append(new_token)
                    skip = True
                else:
                    new_word.append(word[i])
            new_pre_token_cnt[tuple(new_word)] += count
        pre_token_cnt = new_pre_token_cnt

    pbar.close()
    return vocab, merges


def preprocess_test(text: str, special_tokens: list[str]) -> defaultdict[tuple[bytes, bytes], int]:
    def word2tuple(word):
        word = list(word.encode('utf-8'))
        word = [bytes([b]) for b in word]
        return tuple(word)
    pre_token_cnt = defaultdict(int)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    chunks = re.split("|".join(map(re.escape, special_tokens)), text)
    for chunk in chunks:
        for m in re.finditer(PAT, chunk):
            word = m.group(0)
            pre_token_cnt[word2tuple(word)] += 1
    return pre_token_cnt

if __name__ == "__main__":
    DATA_PATH = pathlib.Path(__file__).resolve().parent.parent / "data"
    # INPUT_PATH = os.path.join(DATA_PATH, "owt_train.txt")
    INPUT_PATH = os.path.join(DATA_PATH, "TinyStoriesV2-GPT4-train.txt")
    TOKENIZER_DIR = pathlib.Path(__file__).resolve().parent.parent/ "tokenizer"
    # VOCAB_PATH = os.path.join(TOKENIZER_DIR, "owt_bpe_vocab.pkl")
    # MERGES_PATH = os.path.join(TOKENIZER_DIR, "owt_bpe_merges.pkl")
    VOCAB_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_vocab.pkl")
    MERGES_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_merges.pkl")
    special_tokens = ["<|endoftext|>"]

    vocab, merges = train_tokenizer_bpe(
        input_path=INPUT_PATH,
        vocab_size=10000,
        special_tokens=special_tokens
    )
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(vocab, f)
    with open(MERGES_PATH, "wb") as f:
        pickle.dump(merges, f)

    # 统计最长 token
    longest_token = max(vocab.values(), key=len)
    print("最长token:", longest_token, "长度:", len(longest_token))