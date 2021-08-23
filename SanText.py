
import random
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from functools import partial
from multiprocessing import Pool, cpu_count


def cal_probability(word_embed_1, word_embed_2, epsilon=2.0):
    distance = euclidean_distances(word_embed_1, word_embed_2)
    sim_matrix = -distance
    prob_matrix = softmax(epsilon * sim_matrix / 2, axis=1)
    return prob_matrix


def SanText_init(prob_matrix_init,):
    global prob_matrix
    prob_matrix = prob_matrix_init

def SanText(doc):
    new_doc = []
    for token in doc:
        sampling_prob = prob_matrix[token]
        sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
        new_doc.append(sampling_index[0])
    return new_doc


def SanText_plus_init(prob_matrix_init, word2id_init, sword2id_init, all_words_init, p_init, tokenizer_init):
    global prob_matrix
    global word2id
    global sword2id
    global id2sword
    global all_words
    global p
    global tokenizer

    prob_matrix = prob_matrix_init
    word2id = word2id_init
    sword2id=sword2id_init

    id2sword = {v: k for k, v in sword2id.items()}

    all_words = all_words_init
    p=p_init
    tokenizer=tokenizer_init

def SanText_plus(doc):
    new_doc = []
    for word in doc:
        if word in word2id:
            # In-vocab
            if word in sword2id:
                #Sensitive Words
                index = word2id[word]
                sampling_prob = prob_matrix[index]
                sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
                new_doc.append(id2sword[sampling_index[0]])
            else:
                #Non-sensitive words
                flip_p=random.random()
                if flip_p<=p:
                    #sample a word from Vs based on prob matrix
                    index = word2id[word]
                    sampling_prob = prob_matrix[index]
                    sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
                    new_doc.append(id2sword[sampling_index[0]])
                else:
                    #keep as the original
                    new_doc.append(word)
        else:
            #Out-of-Vocab words
            sampling_prob = 1 / len(all_words) * np.ones(len(all_words), )
            sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
            new_doc.append(all_words[sampling_index[0]])

    new_doc = " ".join(new_doc)
    return new_doc


def get_sanitized_doc(docs, embedding_matrix, epsilon=2.0, threads=12):
    threads = min(threads, cpu_count())

    prob_matrix=cal_probability(embedding_matrix, embedding_matrix, epsilon=epsilon,)

    with Pool(threads, initializer=SanText_init, initargs=(prob_matrix,)) as p:
        annotate_ = partial(
            SanText,
        )
        results = list(
            tqdm(
                p.imap(annotate_, docs, chunksize=32),
                total=len(docs),
                desc="Sanitize docs using SanText",
            )
        )
        p.close()
    
    return results
