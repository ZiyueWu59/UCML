import os
import torch
import torchtext
import numpy as np
from torch.functional import F
import pickle
import torch.nn as nn
from utils.categories_v2 import PKU_vidvrd_categories, vidvrd_pred_categories


vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
vocab.itos.extend(['<unk>'])
vocab.stoi['<unk>'] = vocab.vectors.shape[0]
vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)

def embedding(sentence, vocabs=[], embedders=[]):
    if len(vocabs) == 0:
        vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
        vocab.itos.extend(['<unk>'])
        vocab.stoi['<unk>'] = vocab.vectors.shape[0]
        vocab.vectors = torch.cat(
            [vocab.vectors, torch.zeros(1, vocab.dim)],
            dim=0
        )
        vocabs.append(vocab)

    if len(embedders) == 0:
        embedder = torch.nn.Embedding.from_pretrained(vocab.vectors)
        embedders.append(embedder)

    vocab, embedder = vocabs[0], embedders[0]
    word_idxs = torch.tensor([vocab.stoi.get(w.lower(), 400000) \
                              for w in sentence.split()], dtype=torch.long)

    return embedder(word_idxs)


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=300, feat_dim=512, num_layers=3, bidirectional=True):
        super().__init__()
        
        if bidirectional:
            feat_dim //= 2
        self.lstm = nn.LSTM(
            input_dim, feat_dim, num_layers=num_layers,
            bidirectional=bidirectional, batch_first=True
        )
        if bidirectional:
            feat_dim *= 2
        self.fc_full = nn.Linear(feat_dim, feat_dim)  

    def forward(self, input_embs): 

        self.lstm.flatten_parameters()
        queries = self.lstm(input_embs)[0]
        queries_start = queries[range(queries.size(0)),0]

        prompt_wordlens = torch.tensor([queries.shape[1]], dtype=torch.int16).expand(queries.size(0)).to(queries.device)
        queries_end = queries[range(queries.size(0)), prompt_wordlens.long() - 1]
        full_queries = (queries_start + queries_end) / 2

        return self.fc_full(full_queries)    


class PromptLearners(nn.Module):

    def __init__(self, n_ctx=16, input_dim=300, feat_dim=512, all_emb_path=None, lstm_input=300, lstm_feat_dim=512):
        super().__init__()

        self.all_emb_path = 'UCML/vidvrd_relation.pkl' if all_emb_path is None else all_emb_path
        relations = pickle.load(open(self.all_emb_path, 'rb'))
        self.AllEmb = nn.Parameter(relations['embedding'], requires_grad=False)

        self.n_ctx = n_ctx
        self.n_cls = 36 
        ctx_vectors = torch.empty(n_ctx, input_dim, dtype=torch.float) 
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        prompt_prefix = " ".join(["X"] * n_ctx)

        categories_set = PKU_vidvrd_categories[1:]
        categories_set = [name.replace('_', ' ') for name in categories_set]
        self.prompt_lens = [len(name.split(' '))+16+1 for name in categories_set]
        prompt = [prompt_prefix + " " + "." for name in categories_set]


        embeddings = []
        for idx in range(len(prompt)):
            p = prompt[idx]
            emb = embedding(p)
            cat_emb = self.AllEmb[idx+1]
            new_emb = torch.cat([emb[:-1], cat_emb.unsqueeze(0), emb[-1].unsqueeze(0)], dim=0)
            embeddings.append(new_emb.unsqueeze(0))
        embeddings = torch.cat(embeddings)

        self.register_buffer("token_prefix", embeddings[:, :1, :])  
        self.register_buffer("token_suffix", embeddings[:, 1 + n_ctx :, :]) 

        self.lstm = LSTMEncoder(input_dim=lstm_input, feat_dim=lstm_feat_dim)

    def forward(self, id_list):

        noBG_id = id_list - 1

        ctx = self.ctx
        ctx = ctx.unsqueeze(0).expand(len(noBG_id), -1, -1) 

        prefix = self.token_prefix[noBG_id]
        suffix = self.token_suffix[noBG_id]

        prompts = torch.cat(
            [
                prefix, 
                ctx,    
                suffix, 
            ],
            dim=1,
        )
       
        encode_prompt = self.lstm(prompts)
        return encode_prompt

