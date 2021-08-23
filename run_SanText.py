import argparse
import torch
import random
import numpy as np
import logging
import os
logger = logging.getLogger(__name__)
from tqdm import tqdm
from scipy.special import softmax
from functools import partial
from multiprocessing import Pool, cpu_count
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from utils import get_vocab_SST2, get_vocab_CliniSTS, get_vocab_QNLI, word_normalize
from spacy.lang.en import English
from transformers import BertTokenizer, BertForMaskedLM
from SanText import SanText_plus,SanText_plus_init



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def cal_probability(word_embed_1, word_embed_2, epsilon=2.0):
    distance = euclidean_distances(word_embed_1, word_embed_2)
    sim_matrix = -distance
    prob_matrix = softmax(epsilon * sim_matrix / 2, axis=1)
    return prob_matrix


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default="./data/SST-2/",
        type=str,
        help="The input dir"
    )

    parser.add_argument(
        "--bert_model_path",
        default="bert-base-uncased",
        type=str,
        help="bert model name or path. leave it bank if you are using Glove"
    )

    parser.add_argument(
        "--output_dir",
        default="./output_SanText/QNLI/",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--word_embedding_path",
        default='./data/glove.840B.300d.txt',
        type=str,
        help="The pretrained word embedding path. leave it blank if you are using BERT",
    )

    parser.add_argument(
        "--word_embedding_size",
        default=300,
        type=int,
        help="The pretrained word embedding size. leave it blank if you are using BERT",
    )

    parser.add_argument(
        '--method',
        choices=['SanText', 'SanText_plus'],
        default='SanText_plus',
        help='Sanitized method'
    )

    parser.add_argument(
        '--embedding_type',
        choices=['glove', 'bert'],
        default='glove',
        help='embedding used for sanitization'
    )

    parser.add_argument('--task',
                        choices=['CliniSTS', "SST-2", "QNLI"],
                        default='SST-2',
                        help='NLP eval tasks')

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--epsilon", type=float, default=15, help="privacy parameter epsilon")
    parser.add_argument("--p", type=float, default=0.2, help="SanText+: probability of non-sensitive words to be sanitized")

    parser.add_argument("--sensitive_word_percentage", type=float, default=0.5,
                        help="SanText+: how many words are treated as sensitive")

    parser.add_argument("--threads", type=int, default=12, help="number of processors")

    args = parser.parse_args()

    set_seed(args)

    logging.basicConfig(
        format="%(asctime)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info("Running method: %s, task: %s,  epsilon = %s, random_seed: %d" % (
    args.method, args.task, args.epsilon, args.seed))

    if args.method == "SanText":
        args.sensitive_word_percentage = 1.0
        args.output_dir = os.path.join(args.output_dir, "eps_%.2f" % args.epsilon)
    else:
        args.output_dir = os.path.join(args.output_dir, "eps_%.2f" % args.epsilon, "sword_%.2f_p_%.2f"%(args.sensitive_word_percentage,args.p))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("Building Vocabulary...")

    if args.embedding_type=="glove":
        tokenizer = English()
        tokenizer_type="word"
    else:
        tokenizer  = BertTokenizer.from_pretrained(args.bert_model_path)
        tokenizer_type = "subword"
    if args.task == "SST-2":
        vocab = get_vocab_SST2(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
    elif args.task == "CliniSTS":
        vocab = get_vocab_CliniSTS(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
    elif args.task == "QNLI":
        vocab = get_vocab_QNLI(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
    else:
        raise NotImplementedError

    sensitive_word_count = int(args.sensitive_word_percentage * len(vocab))
    words = [key for key, _ in vocab.most_common()]
    sensitive_words = words[-sensitive_word_count - 1:]

    sensitive_words2id = {word: k for k, word in enumerate(sensitive_words)}
    logger.info("#Total Words: %d, #Sensitive Words: %d" % (len(words),len(sensitive_words2id)))

    sensitive_word_embed = []
    all_word_embed=[]

    word2id = {}
    sword2id = {}
    sensitive_count = 0
    all_count = 0
    if args.embedding_type == "glove":
        num_lines = sum(1 for _ in open(args.word_embedding_path))
        logger.info("Loading Word Embedding File: %s" % args.word_embedding_path)

        with open(args.word_embedding_path) as f:
            # Skip first line if of form count/dim.
            line = f.readline().rstrip().split(' ')
            if len(line) != 2:
                f.seek(0)
            for row in tqdm(f, total=num_lines - 1):
                content = row.rstrip().split(' ')
                cur_word=word_normalize(content[0])
                if cur_word in vocab and cur_word not in word2id:
                    word2id[cur_word] = all_count
                    all_count += 1
                    emb=[float(i) for i in content[1:]]
                    all_word_embed.append(emb)
                    if cur_word in sensitive_words2id:
                        sword2id[cur_word] = sensitive_count
                        sensitive_count += 1
                        sensitive_word_embed.append(emb)
                assert len(word2id)==len(all_word_embed)
                assert len(sword2id) == len(sensitive_word_embed)
            f.close()
    else:
        logger.info("Loading BERT Embedding File: %s" % args.bert_model_path)
        model=BertForMaskedLM.from_pretrained(args.bert_model_path)
        embedding_matrix = model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()

        for cur_word in tokenizer.vocab:
            if cur_word in vocab and cur_word not in word2id:
                word2id[cur_word] = all_count
                emb = embedding_matrix[tokenizer.convert_tokens_to_ids(cur_word)]
                all_word_embed.append(emb)
                all_count += 1

                if cur_word in sensitive_words2id:
                    sword2id[cur_word] = sensitive_count
                    sensitive_count += 1
                    sensitive_word_embed.append(emb)
            assert len(word2id) == len(all_word_embed)
            assert len(sword2id) == len(sensitive_word_embed)

    all_word_embed=np.array(all_word_embed, dtype='f')
    sensitive_word_embed = np.array(sensitive_word_embed, dtype='f')

    logger.info("All Word Embedding Matrix: %s" % str(all_word_embed.shape))
    logger.info("Sensitive Word Embedding Matrix: %s" % str(sensitive_word_embed.shape))

    logger.info("Calculating Prob Matrix for Exponential Mechanism...")
    prob_matrix = cal_probability(all_word_embed,sensitive_word_embed, args.epsilon)

    threads = min(args.threads, cpu_count())

    for file_name in ['train.tsv','dev.tsv']:
        data_file = os.path.join(args.data_dir, file_name)
        out_file = open(os.path.join(args.output_dir, file_name), 'w')
        logger.info("Processing file: %s. Will write to: %s" % (data_file,os.path.join(args.output_dir, file_name)))

        num_lines = sum(1 for _ in open(data_file))
        with open(data_file, 'r') as rf:
            # header
            header = next(rf)
            out_file.write(header)
            labels = []
            docs = []
            if args.task == "SST-2":
                for line in tqdm(rf, total=num_lines - 1):
                    content = line.strip().split("\t")
                    text = content[0]
                    label = int(content[1])
                    if args.embedding_type == "glove":
                        doc = [token.text for token in tokenizer(text)]
                    else:
                        doc = tokenizer.tokenize(text)
                    docs.append(doc)
                    labels.append(label)
            elif args.task == "CliniSTS":
                for line in tqdm(rf, total=num_lines - 1):
                    content = line.strip().split("\t")
                    text1 = content[7]
                    text2 = content[8]
                    label = content[-1]
                    if args.embedding_type == "glove":
                        doc1 = [token.text for token in tokenizer(text1)]
                        doc2 = [token.text for token in tokenizer(text2)]
                    else:
                        doc1 = tokenizer.tokenize(text1)
                        doc2 = tokenizer.tokenize(text2)
                    docs.append(doc1)
                    docs.append(doc2)
                    labels.append(label)
            elif args.task == "QNLI":
                for line in tqdm(rf, total=num_lines - 1):
                    content = line.strip().split("\t")
                    text1 = content[1]
                    text2 = content[2]
                    label = content[-1]
                    if args.embedding_type == "glove":
                        doc1 = [token.text for token in tokenizer(text1)]
                        doc2 = [token.text for token in tokenizer(text2)]
                    else:
                        doc1 = tokenizer.tokenize(text1)
                        doc2 = tokenizer.tokenize(text2)

                    docs.append(doc1)
                    docs.append(doc2)
                    labels.append(label)

            rf.close()

        with Pool(threads, initializer=SanText_plus_init, initargs=(prob_matrix, word2id, sword2id, words, args.p, tokenizer)) as p:
            annotate_ = partial(
                SanText_plus,
            )
            results = list(
                tqdm(
                    p.imap(annotate_, docs, chunksize=32),
                    total=len(docs),
                    desc="Sanitize docs using SanText",
                )
            )
            p.close()

        logger.info("Saving ...")

        if args.task == "SST-2":
            for i, predicted_text in enumerate(results):
                write_content = predicted_text + "\t" + str(labels[i]) + "\n"
                out_file.write(write_content)
        elif args.task == "CliniSTS":
            assert len(results) / 2 == len(labels)
            for i in range(len(labels)):
                predicted_text1 = results[i*2]
                predicted_text2 = results[i*2+1]
                write_content = str(i) + "\t" + "none\t" * 6 + predicted_text1 + "\t" + predicted_text2 + "\t" + str(
                    labels[i]) + "\n"
                out_file.write(write_content)
        elif args.task == "QNLI":
            assert len(results) / 2 == len(labels)
            for i in range(len(labels)):
                predicted_text1 = results[i*2]
                predicted_text2 = results[i*2+1]
                write_content = str(i) + "\t" + predicted_text1 + "\t" + predicted_text2 + "\t" + str(
                    labels[i]) + "\n"
                out_file.write(write_content)

        out_file.close()



if __name__ == "__main__":
    main()
