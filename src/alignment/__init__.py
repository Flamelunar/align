from dataclasses import dataclass
from typing import Dict
import numpy as np
import logging
from tqdm.auto import tqdm
import math
from sklearn.metrics.pairwise import cosine_similarity
import os
from pathlib import Path
import fasttext
import fasttext.util
# import tempfile
# from datasets import load_dataset
# import nltk
# from functools import partial
# import multiprocessing
from scipy.linalg import orthogonal_procrustes
from gensim.models import Word2Vec
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
__version__ = "0.0.4"

CACHE_DIR = Path("/home/ljj/model/bin").resolve()

def softmax(x, axis=-1):
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


class WordEmbedding:

    def __init__(self, model):
        self.model = model

        if isinstance(model, fasttext.FastText._FastText):
            self.kind = "fasttext"
        elif isinstance(model, Word2Vec):
            self.kind = "word2vec"
        else:
            raise ValueError(
                f"{model} seems to be neither a fastText nor Word2Vec model."
            ) 

    def has_subword_info(self):
        return self.kind == "fasttext"

    def get_words_and_freqs(self):
        if self.kind == "fasttext":
            return self.model.get_words(include_freq=True, on_unicode_error="ignore")
        elif self.kind == "word2vec":
            return (self.model.wv.index_to_key, self.model.wv.expandos["count"])

    def get_dimension(self):
        if self.kind == "fasttext":
            return self.model.get_dimension()
        elif self.kind == "word2vec":
            return self.model.wv.vector_size

    def get_word_vector(self, word):
        if self.kind == "fasttext":
            return self.model.get_word_vector(word)
        elif self.kind == "word2vec":
            return self.model.wv[word]

    def get_word_id(self, word):
        if self.kind == "fasttext":
            return self.model.get_word_id(word)
        elif self.kind == "word2vec":
            return self.model.wv.key_to_index.get(word, -1)


def get_subword_embeddings_in_word_embedding_space(
    subwords,counter,subword_idxs, tokenizer, model, max_n_word_vectors=None, use_subword_info=True, verbose=True
):

    one_batch_id_num = subword_idxs.numel() 
    b = subword_idxs.reshape(-1)
    one_batch_id = b.tolist()

    words, freqs = model.get_words_and_freqs()  

    if max_n_word_vectors is None:
        max_n_word_vectors = len(words) 
    sources = {} 
    embs_matrix = np.zeros((one_batch_id_num, model.get_dimension())) 

    if use_subword_info:
        if not model.has_subword_info():
            raise ValueError("Can not use subword info of model without subword info!")

        for id, i in zip(one_batch_id, range(one_batch_id_num)):
            
            token = tokenizer.decode(id).strip() 
            embs_matrix[i] = model.get_word_vector(token)  #1980*300
   
    else:   
        embs = {}
        v = []
        vocab = tokenizer.get_vocab()

        for i in one_batch_id:
            token = tokenizer.decode(i).strip()
            if token not in vocab and token!= '':
                token = '<unk>'  
            if token == '':
                token = '▁'      
            v.append(vocab[token])

        special_tokens = tokenizer.special_tokens_map.values()
        for spi in special_tokens:
            v.append(vocab[spi]) 
            
        embs = {value: [] for value in v} 
        max_n_word_vectors = len(v) 
        for i, word in tqdm(       
            enumerate(words[:max_n_word_vectors]),
            total=max_n_word_vectors,
            disable=not verbose,
        ):
            for tokenized in [
                
                tokenizer.encode(word, add_special_tokens=False),
                tokenizer.encode(" " + word, add_special_tokens=False),
            ]:
                for token_id in set(tokenized):
                    if token_id not in embs:
                        continue
                    embs[token_id].append(i)
        for i in range(len(embs_matrix)):
            if i not in embs:
                continue
            else:
                if len(embs[i]) == 0 : 
                    continue
            

            weight = np.array([freqs[idx] for idx in embs[i]]) 
            weight = weight / weight.sum()  

            vectors = [model.get_word_vector(words[idx]) for idx in embs[i]] 

            sources[tokenizer.convert_ids_to_tokens([i])[0]] = embs[i]  
            embs_matrix[i] = (np.stack(vectors) * weight[:, np.newaxis]).sum(axis=0)
    return embs_matrix, sources



def load_embeddings(identifier: str, verbose=True):
    
    if os.path.exists(identifier):
        path = Path(identifier)
    else:
        logging.info(
            f"Identifier '{identifier}' does not seem to be a path (file does not exist). Interpreting as language code."
        )

        path = CACHE_DIR / f"cc.{identifier}.300.bin"

    return fasttext.load_model(str(path))  


def create_target_embeddings(
    get_vocab,
    source_subwords,
    targetr_subwords,
    source_counter,
    target_counter,
    source_subword_idxs,
    target_subword_idxs,
    source_subword_embeddings,
    target_subword_embeddings,
    source_tokenizer,
    target_tokenizer,
    source_matrix,
    neighbors=10,
    temperature=0.1,
    verbose=True,
):
    
    def get_n_closest(token_id, similarities, top_k):
        if (target_subword_embeddings[token_id] == 0).all():     
          return None
        
        best_indices = np.argpartition(similarities, -top_k)[-top_k:]
        best_tokens = source_tokenizer.convert_ids_to_tokens(best_indices)
        
        best = sorted(
            [
                (token, similarities[idx])
                for token, idx in zip(best_tokens, best_indices)
            ],
            key=lambda x: -x[1],
        )

        return best
    
    s_one_batch_id_num = source_subword_idxs.numel()  
    s = source_subword_idxs.reshape(-1)
    s_one_batch_id = s.tolist()
    t = target_subword_idxs.reshape(-1)
    t_one_batch_id = t.tolist()
    source_vocab = get_vocab
    target_vocab = get_vocab

    target_matrix = np.zeros(
        (target_subword_idxs.numel(), source_matrix.shape[1]), dtype=source_matrix.dtype  
    )
    
    mean, std = (
        source_matrix.mean(0),
        source_matrix.std(0),
    )
    random_fallback_matrix = np.random.RandomState(1234).normal(
        mean, std, target_matrix.shape # 1254*768
    )

    batch_size = len(target_matrix)   

    n_matched = 0
    
    not_found = []
     
    sources = {}
      
    for i in tqdm(
        range(int(math.ceil(len(target_matrix) / batch_size))), disable=not verbose
        
    ):
        start, end = (
            i * batch_size,
            min((i + 1) * batch_size, len(target_matrix)),
        )
        similarities = cosine_similarity( 
            target_subword_embeddings[start:end], 
            source_subword_embeddings
        ) 
        for token_id in range(start, end):
            closest = get_n_closest(token_id, similarities[token_id - start], neighbors)
            s = similarities[token_id - start]
            # print(s)

            if closest is not None:
                tokens, sims = zip(*closest)
                weights = softmax(np.array(sims) / temperature, 0)
                
                sources[target_tokenizer.convert_ids_to_tokens(token_id)] = (
                    tokens,
                    weights,
                    sims,
                )

                emb = np.zeros(target_matrix.shape[1]) 

                for i, close_token in enumerate(tokens):
                    emb += source_matrix[source_vocab[close_token]] * weights[i]  

                target_matrix[token_id] = emb # 

                n_matched += 1
            else:
                
                target_matrix[token_id] = random_fallback_matrix[token_id]
                not_found.append(target_tokenizer.convert_ids_to_tokens([token_id])[0])
    special_tokens_map_values = source_tokenizer.special_tokens_map.values()
    for token in special_tokens_map_values:
        if isinstance(token, str):
            token = [token]
        for t in token:
            if t in target_vocab and t != "<mask>":
                target_matrix[target_vocab[t]] = source_matrix[
                    source_vocab[t]
                ]

    logging.info(
        f"Matching token found for {n_matched} of {len(target_matrix)} tokens."
    )
    print(f"Matching token found for {n_matched} of {len(target_matrix)} tokens.")
    
    return target_matrix, not_found, sources  


@dataclass
class AlignmentInfo:
    source_subword_sources: Dict
    target_subword_sources: Dict
    sources: Dict
    not_found: Dict


# Alignment
class Alignment:
    def _compute_align_matrix_from_dictionary(
        self, source_embeddings, target_embeddings, dictionary
    ):
        
        correspondences = [] 

        for source_word, target_word in dictionary:
            for trg_w in (source_word, source_word.lower(), source_word.title()):
                for src_w in (target_word, target_word.lower(), target_word.title()):
                    src_id = source_embeddings.get_word_id(src_w)
                    trg_id = target_embeddings.get_word_id(trg_w)
                  
                    if src_id != -1 and trg_id != -1: 
                        correspondences.append(
                            [   
                                source_embeddings.get_word_vector(src_w),
                                target_embeddings.get_word_vector(trg_w),
                            ]
                        )
        correspondences = np.array(correspondences) 

        align_matrix, _ = orthogonal_procrustes(
            
            correspondences[:, 0], correspondences[:, 1] 
        )      
        print(f"align_matrix has got, its shape is {align_matrix.shape}")
        return align_matrix

    def __init__(
        self,
        source_embeddings,
        target_embeddings,
        align_strategy,
        bilingual_dictionary,  
    ):
        """
        Args:
            source_embeddings: fastText model or gensim Word2Vec model in the source language.

            target_embeddings: fastText model or gensim Word2Vec model in the source language.
            align_strategy: either of "bilingual_dictionary" or `None`.
                - If `None`, embeddings are treated as already aligned.
                - If "bilingual dictionary", a bilingual dictionary must be passed
                    which will be used to align the embeddings using the Orthogonal Procrustes method.
            bilingual_dictionary: path to a bilingual dictionary. The dictionary must be of the form
                ```
                english_word1 \t target_word1\n
                english_word2 \t target_word2\n
                ...
                english_wordn \t target_wordn\n
                ```
        """
        print("------Wechsel initialization is running------")
        source_embeddings = WordEmbedding(source_embeddings)
        target_embeddings = WordEmbedding(target_embeddings)

        min_dim = min(
            source_embeddings.get_dimension(), target_embeddings.get_dimension()
        )  # min_dim=300
        if source_embeddings.get_dimension() != min_dim:
            fasttext.util.reduce_model(source_embeddings.model, min_dim)
        if target_embeddings.get_dimension() != min_dim:
            fasttext.util.reduce_model(source_embeddings.model, min_dim)
        print(f"source and target dimension's min_dim={min_dim}")

        if align_strategy == "bilingual_dictionary":
            if bilingual_dictionary is None:
                
                raise ValueError(
                    "`bilingual_dictionary` must not be `None` if `align_strategy` is 'bilingual_dictionary'."
                )

            dictionary = []
            # 遍历字典
            for line in open(bilingual_dictionary):
                line = line.strip()
                try:
                    source_word, target_word = line.split("\t")
                except ValueError:
                    source_word, target_word = line.split()
                dictionary.append((source_word, target_word))

            align_matrix = self._compute_align_matrix_from_dictionary(
                source_embeddings, target_embeddings, dictionary
            ) 
            self.source_transform = lambda matrix: matrix @ align_matrix
            self.target_transform = lambda x: x
            
        elif align_strategy is None:
            self.source_transform = lambda x: x
            self.target_transform = lambda x: x
        else:
            raise ValueError(f"Unknown align strategy: {align_strategy}.")
        self.source_embeddings = source_embeddings
        self.target_embeddings = target_embeddings
        print("------Wechsel initialization is complete!------")

    def apply(
        self,
        get_vocab,
        source_subwords,
        targetr_subwords,
        source_counter,
        target_counter,
        source_subword_idxs,
        target_subword_idxs,
        source_tokenizer,
        target_tokenizer,
        source_matrix,          
        use_subword_info=True,
        max_n_word_vectors=None,
        neighbors=10,
        temperature=0.1,
    ):
         
        """
        Applies Alignment to initialize an embedding matrix.
        Args:
            source_tokenizer: T^s, the tokenizer in the source language.
            target_tokenizer: T^t, the tokenizer in the target language.
            source_matrix: E^s, the embeddings in the source language.
            use_subword_info: Whether to use fastText subword information. Default true.
            max_n_word_vectors: Maximum number of vectors to consider (only relevant if `use_subword_info` is False).

        Returns:
            target_matrix: The embedding matrix for the target tokenizer.
            info: Additional info about word sources, etc.
        """
        (
            source_subword_embeddings,  
            source_subword_sources,
        ) = get_subword_embeddings_in_word_embedding_space(
            source_subwords,
            source_counter,
            source_subword_idxs,
            source_tokenizer,
            self.source_embeddings,
            use_subword_info=use_subword_info,
            max_n_word_vectors=max_n_word_vectors,
        )
        (
            target_subword_embeddings,   #(1254,300)
            target_subword_sources,
        ) = get_subword_embeddings_in_word_embedding_space(
            targetr_subwords,
            target_counter,
            target_subword_idxs,
            target_tokenizer,
            self.target_embeddings,
            use_subword_info=use_subword_info,
            max_n_word_vectors=max_n_word_vectors,
        )
        soriginal_embeddings = source_subword_embeddings
        toriginal_embeddings = target_subword_embeddings

        source_subword_embeddings = self.source_transform(source_subword_embeddings)  
        target_subword_embeddings = self.target_transform(target_subword_embeddings)
        source_subword_embeddings /= (
            np.linalg.norm(source_subword_embeddings, axis=1)[:, np.newaxis] + 1e-8
        )
        target_subword_embeddings /= (
            np.linalg.norm(target_subword_embeddings, axis=1)[:, np.newaxis] + 1e-8
        )
        
        target_matrix, not_found, sources = create_target_embeddings(
            get_vocab,
            source_subwords,  # len()=1980 tuple
            targetr_subwords, # len()=1254 tuple
            source_counter,
            target_counter,   
            source_subword_idxs, # [55,36] tenosr
            target_subword_idxs,  # [57,22] tensor
            source_subword_embeddings, # (1980,300) array
            target_subword_embeddings, # (1254,300) array
            source_tokenizer,
            target_tokenizer,
            source_matrix.copy(),  #(1980,768) array
            neighbors=neighbors,   # 10
            temperature=temperature,  # 0.1
        )
        
        
        return target_matrix, AlignmentInfo(
            source_subword_sources=source_subword_sources,
            target_subword_sources=target_subword_sources,
            sources=sources,
            not_found=not_found,
        )
