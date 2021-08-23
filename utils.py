from tqdm import tqdm
import os
import unicodedata
from collections import Counter

def word_normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def get_vocab_SST2(data_dir,tokenizer,tokenizer_type="subword"):
    vocab=Counter()
    for split in ['train','dev']:
        data_file_path=os.path.join(data_dir,split+".tsv")
        num_lines = sum(1 for _ in open(data_file_path))
        with open(data_file_path, 'r') as csvfile:
                next(csvfile)
                for line in tqdm(csvfile,total=num_lines-1):
                    line=line.strip().split("\t")
                    text = line[0]
                    if tokenizer_type=="subword":
                        tokenized_text = tokenizer.tokenize(text)
                    elif tokenizer_type=="word":
                        tokenized_text = [token.text for token in tokenizer(text)]
                    for token in tokenized_text:
                        vocab[token]+=1
    if tokenizer_type == "subword":
        for token in tokenizer.vocab:
            vocab[token]+=1
    return vocab

def get_vocab_CliniSTS(data_dir,tokenizer,tokenizer_type="subword"):
    vocab=Counter()
    for split in ['train','dev']:
        data_file_path=os.path.join(data_dir,split+".tsv")
        num_lines = sum(1 for _ in open(data_file_path))
        with open(data_file_path, 'r') as csvfile:
            next(csvfile)
            for line in tqdm(csvfile,total=num_lines-1):
                line = line.strip().split("\t")
                text = line[7] + " " + line[8]
                if tokenizer_type=="subword":
                    tokenized_text = tokenizer.tokenize(text)
                elif tokenizer_type=="word":
                    tokenized_text = [token.text for token in tokenizer(text)]
                for token in tokenized_text:
                    vocab[token]+=1
    if tokenizer_type == "subword":
        for token in tokenizer.vocab:
            vocab[token]+=1
    return vocab

def get_vocab_QNLI(data_dir,tokenizer,tokenizer_type="subword"):
    vocab=Counter()
    for split in ['train','dev']:
        data_file_path=os.path.join(data_dir,split+".tsv")
        num_lines = sum(1 for _ in open(data_file_path))
        with open(data_file_path, 'r') as csvfile:
            next(csvfile)
            for line in tqdm(csvfile,total=num_lines-1):
                line = line.strip().split("\t")
                text = line[1] + " " + line[2]
                if tokenizer_type=="subword":
                    tokenized_text = tokenizer.tokenize(text)
                elif tokenizer_type=="word":
                    tokenized_text = [token.text for token in tokenizer(text)]
                for token in tokenized_text:
                    vocab[token]+=1
    if tokenizer_type == "subword":
        for token in tokenizer.vocab:
            vocab[token]+=1
    return vocab



