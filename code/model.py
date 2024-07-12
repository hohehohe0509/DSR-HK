import datetime
import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from scipy.sparse import coo_matrix,csr_matrix
from time import time
import random
from numba import jit
import heapq
import dgl
from conv import myGATConv, DropLearner
from tqdm import tqdm

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

class HyperConv(Module):
    def __init__(self, layers,dataset,emb_size=112):
        super(HyperConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.dataset = dataset

    def forward(self, adjacency, embedding):
        item_embeddings = embedding
        item_embedding_layer0 = item_embeddings
        final = [item_embedding_layer0]
        for i in range(self.layers):
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings)
            final.append(item_embeddings)
        item_embeddings = torch.sum(torch.stack(final),dim=0)/(self.layers+1)
        return item_embeddings
    
class SessConv(Module):
    def __init__(self, layers, batch_size, emb_size=112):
        super(SessConv, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.layers = layers
        # self.w_sess = {}

        self.W_q = nn.Parameter(torch.Tensor(self.emb_size, self.emb_size))
        self.W_k = nn.Parameter(torch.Tensor(self.emb_size, self.emb_size))
        self.W_init = nn.Parameter(torch.Tensor(self.emb_size, 2*self.emb_size))
        self.w_sess = nn.ModuleList([nn.Linear(self.emb_size, self.emb_size, bias=False) for i in range(self.layers)])

    def SR_IEM(self, item_embedding, session_item, session_len, mask):
        d = trans_to_cuda(torch.Tensor([self.emb_size]))

        seq_h = torch.cuda.FloatTensor(self.batch_size, self.emb_size).fill_(0)
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        cl_embed = torch.cat([zeros, item_embedding], 0)
        seq  = cl_embed[session_item]

        Q = torch.sigmoid(torch.matmul(seq,self.W_q))
        K = torch.sigmoid(torch.matmul(seq,self.W_k))
        C = torch.div(torch.sigmoid(torch.matmul(Q, K.permute(0,2,1))), torch.sqrt(d.view(-1,1)))

        # Set the diagonal of C to 0
        result_matrix = mask[:, None] * mask.unsqueeze(2)
        C = result_matrix * C
        diag = torch.diagonal(C, dim1=1, dim2=2)
        a_diag = torch.diag_embed(diag)
        C = C-a_diag
        
        # Equation (3.10)
        alpha = torch.div(torch.sum(C, 2),(session_len))
        get = lambda i: alpha[i,:session_len[i]]

        # Equation (3.11)
        for i in torch.arange(session_item.shape[0]):
            beta = F.softmax(get(i), 0)
            session_long = torch.matmul(beta,seq[i,:session_len[i]])
            seq_h[i] = session_long

        return seq_h

    def forward(self, item_embedding, D, A, session_item, session_len, mask):
        session_emb = self.SR_IEM(item_embedding, session_item, session_len, mask)
        session = [session_emb]
        DA = torch.mm(D, A).float()

        # Equation (3.12)
        for i in range(self.layers):
            session_emb = trans_to_cuda(self.w_sess[i])(session_emb)
            session_emb = torch.mm(DA, session_emb)
            session.append(F.normalize(session_emb, p=2, dim=-1))
        sess = trans_to_cuda(torch.tensor(np.array([item.cpu().detach().numpy() for item in session])))
        session_emb = torch.sum(sess, 0)/(self.layers+1)
        return session_emb


class COTREC(Module):
    def __init__(self, adjacency, n_node, n_item, opt, num_layers, num_hidden, num_classes,
                 heads, activation, feat_drop, attn_drop, negative_slope, residual, ret_num, n_relations=0, emb_size=100, relation_embSize=100):
        super(COTREC, self).__init__()
        self.emb_size = emb_size
        self.batch_size = opt.batchSize
        self.kg_embSize = opt.kg_embSize
        self.relation_embSize = relation_embSize
        self.n_node = n_node
        self.n_item = n_item
        self.n_relations = n_relations
        self.dataset = opt.dataset
        self.lr = opt.lr
        self.layers = opt.layer
        self.beta = opt.beta
        self.K = opt.K
        self.w_k = 10
        self.num = 5000

        self.kg_gat_layers = nn.ModuleList()
        self.sub_gat_layers = nn.ModuleList()
        self.drop_learner = False
        self.activation = activation
        self.alpha=opt.alpha
        self.epsilon = torch.FloatTensor([1e-12]).cuda()
        self.kg_edge_weight = None
        self.num_layers = num_layers
        self.tau = opt.temperature

        ######
        
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        ######

        
        self.adjacency = adjacency
        
        self.ret_num = ret_num
        self.embedding = nn.Parameter(torch.zeros((self.n_node, self.emb_size)))
        self.cl_embed = nn.Parameter(torch.zeros((self.ret_num, self.kg_embSize)))
        self.kg_embedding = nn.Parameter(torch.zeros((self.n_node, self.kg_embSize)))
        self.pos_len = 200

        self.relation_embed = nn.Parameter(torch.zeros((self.n_relations, self.relation_embSize)))
        self.pos_embedding = nn.Parameter(torch.zeros((self.pos_len, self.emb_size)))

        self.HyperGraph = HyperConv(self.layers,opt.dataset) #多加的
        self.SessGraph = SessConv(self.layers, self.batch_size)
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.emb_size, self.emb_size))
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.w_i = nn.Linear(self.emb_size, self.emb_size)
        self.w_s = nn.Linear(self.emb_size, self.emb_size)
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.glu3 = nn.Linear(self.emb_size, self.emb_size)
        self.linear_transform = nn.Linear(self.emb_size * 2, self.emb_size, bias=True)
        self.W_R = nn.Parameter(torch.Tensor(self.n_relations, self.emb_size, self.relation_embSize))
        

        #用在注意力層
        self.a = nn.Parameter(torch.Tensor(2 * self.emb_size, 1))
        
        # input projection (no residual)
        self.sub_gat_layers.append(myGATConv(self.kg_embSize, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, bias=True, alpha=self.alpha))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.sub_gat_layers.append(myGATConv(num_hidden * heads[l-1],
                 num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, bias=True, alpha=self.alpha))
        # output projection
        self.sub_gat_layers.append(myGATConv(num_hidden * heads[-2],
             num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, bias=True, alpha=self.alpha))


        self.kg_gat_layers.append(myGATConv(self.kg_embSize, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, bias=True, alpha=self.alpha))
        # hidden layers
        for l in range(1, self.num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.kg_gat_layers.append(myGATConv(num_hidden * heads[l-1],
                 num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, bias=True, alpha=self.alpha))
        # output projection
        self.kg_gat_layers.append(myGATConv(num_hidden * heads[-2],
             num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, bias=True, alpha=self.alpha))
        
        self.learner1 = DropLearner(self.kg_embSize, self.kg_embSize)
        self.learner2 = DropLearner(self.kg_embSize, self.kg_embSize, self.kg_embSize)
    
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)  

    # Inverse positional embedding
    def generate_sess_emb(self, item_embedding, session_item, session_len, reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        
        # hs=x^k_s
        hs = torch.div(torch.sum(seq_h, 1), session_len)
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]
        pos_emb = self.pos_embedding[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        # Equation (3.5)
        last_item = item_embedding[reversed_sess_item[:,0]].unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, seq_h], -1), self.w_1)
        nh = torch.tanh(nh)
        # Equation (3.6)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs) +self.glu3(last_item))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1)
        return select

    # No Inverse positional embedding
    def generate_sess_emb_npos(self, item_embedding, session_item, session_len, reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        
        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        hs = torch.div(torch.sum(seq_h, 1), session_len)
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]

        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        last_item = item_embedding[reversed_sess_item[:,0]].unsqueeze(-2).repeat(1, len, 1)
        #####
        nh = seq_h
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs) +self.glu3(last_item))
        #####
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1)
        
        return select

    def example_predicting(self, item_emb, sess_emb):
        x_u = torch.matmul(item_emb, sess_emb.transpose(0,1))
        pos = torch.softmax(x_u, 0)
        return pos

    def SSL_topk(self, anchor, sess_emb, pos, neg):
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 2)

        anchor = F.normalize(anchor + sess_emb, p=2, dim=-1)
        pos = torch.reshape(pos, (self.batch_size, self.K, self.emb_size)) + sess_emb.unsqueeze(1).repeat(1, self.K, 1)
        neg = torch.reshape(neg, (self.batch_size, self.K, self.emb_size)) + sess_emb.unsqueeze(1).repeat(1, self.K, 1)
        pos_score = score(anchor.unsqueeze(1).repeat(1, self.K, 1), F.normalize(pos, p=2, dim=-1))
        neg_score = score(anchor.unsqueeze(1).repeat(1, self.K, 1), F.normalize(neg, p=2, dim=-1))
        pos_score = torch.sum(torch.exp(pos_score / self.tau), 1)
        neg_score = torch.sum(torch.exp(neg_score / self.tau), 1)
        con_loss = -torch.sum(torch.log(pos_score / (pos_score + neg_score)))
        return con_loss

    # Equation (3.16)
    def SSL(self, sess_emb_i, sess_emb_s):
        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:,torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        pos = score(sess_emb_i, sess_emb_s)
        neg1 = score(sess_emb_s, row_column_shuffle(sess_emb_i))
        one = torch.cuda.FloatTensor(neg1.shape[0]).fill_(1)
        con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos))-torch.log(1e-8 + (one - torch.sigmoid(neg1))))
        return con_loss

    def topk_func_random(self, score1,score2, item_emb_I, item_emb_S):
        values, pos_ind_I = score1.topk(self.num, dim=0, largest=True, sorted=True)
        values, pos_ind_S = score2.topk(self.num, dim=0, largest=True, sorted=True)
        pos_emb_I = torch.cuda.FloatTensor(self.K, self.batch_size, self.emb_size).fill_(0)
        pos_emb_S = torch.cuda.FloatTensor(self.K, self.batch_size, self.emb_size).fill_(0)
        neg_emb_I = torch.cuda.FloatTensor(self.K, self.batch_size, self.emb_size).fill_(0)
        neg_emb_S = torch.cuda.FloatTensor(self.K, self.batch_size, self.emb_size).fill_(0)
        for i in torch.arange(self.K):
            pos_emb_S[i] = item_emb_S[pos_ind_I[i]]
            pos_emb_I[i] = item_emb_I[pos_ind_S[i]]
        random_slices = torch.randint(self.K, self.num, (self.K,))  # choose negative items
        for i in torch.arange(self.K):
            neg_emb_S[i] = item_emb_S[pos_ind_I[random_slices[i]]]
            neg_emb_I[i] = item_emb_I[pos_ind_S[random_slices[i]]]
        return pos_emb_I, neg_emb_I, pos_emb_S, neg_emb_S

    def calc_kg_emb(self, g, drop_learn = False):
        all_embed = []
        h = self.kg_embedding
        tmp = h
        edge_weight = None
        reg = 0
        if drop_learn:
            reg, edge_weight = self.learner2(tmp, g, temperature = 0.7)
            self.kg_edge_weight = edge_weight.detach()
        else:
            edge_weight = self.kg_edge_weight
        all_embed.append(tmp)
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.kg_gat_layers[l](g, h, res_attn=res_attn, edge_weight = edge_weight)
            h = h.flatten(1)
            tmp = h
            all_embed.append(tmp)
        # output projection
        logits, _ = self.kg_gat_layers[-1](g, h, res_attn=res_attn, edge_weight = edge_weight)
        logits = logits.mean(1)
        tmp = logits
        all_embed.append(tmp)
        all_embed = torch.sum(torch.stack(all_embed),dim=0)/(self.num_layers+2)
        all_embed = (all_embed / (torch.max(torch.norm(all_embed, dim=1, keepdim=True),self.epsilon)))
        if drop_learn:
            return all_embed, reg
        else:
            return all_embed

    def calc_kg_loss(self, g, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        weight = False
        r_embed = self.relation_embed[r]                 # (kg_batch_size, relation_dim)
        W_r = self.W_R[r]                                # (kg_batch_size, entity_dim, relation_dim)
        embedding = self.calc_kg_emb(g)

        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        item_embedding = torch.cat([zeros, embedding], 0)
        
        h_emb = item_embedding[h]
        pos_t_emb = item_embedding[pos_t]
        neg_t_emb = item_embedding[neg_t]

        r_mul_h = torch.bmm(h_emb.unsqueeze(1), W_r).squeeze(1)             # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_emb.unsqueeze(1), W_r).squeeze(1)     # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_emb.unsqueeze(1), W_r).squeeze(1)     # (kg_batch_size, relation_dim)

        # Equation (3.20)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)     # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)     # (kg_batch_size)
        aug_edge_weight = 1
        if weight:
            emb = self.kg_embedding
            emb = (emb / (torch.max(torch.norm(emb, dim=1, keepdim=True),self.epsilon)))
            _, aug_edge_weight = self.learner2.get_weight(emb[h], emb[pos_t], temperature = 0.7)
            
        # Equation (3.21)
        base_loss = (aug_edge_weight * F.softplus(-neg_score + pos_score)).mean()
        return base_loss

    def train_loss(self, sessionID, session_item, session_len, D, A, reversed_sess_item, mask, epoch, tar, diff_mask, kg):
        item_embeddings_i = self.HyperGraph(self.adjacency, self.embedding)
        item_embeddings_kg, reg_kg = self.calc_kg_emb(kg, True)
        sess_emb_s = self.SessGraph(self.embedding, D, A, session_item, session_len, mask)

        sess_emb_i = self.generate_sess_emb(item_embeddings_i, session_item, session_len, reversed_sess_item, mask)
        sess_emb_kg = self.generate_sess_emb(item_embeddings_kg, session_item, session_len,reversed_sess_item, mask)
        
        sess_emb_i = self.w_k * F.normalize(sess_emb_i, dim=-1, p=2)
        sess_emb_kg = self.w_k * F.normalize(sess_emb_kg, dim=-1, p=2)
        item_embeddings_i = F.normalize(item_embeddings_i, dim=-1, p=2)
        item_embeddings_kg = F.normalize(item_embeddings_kg, dim=-1, p=2)
        
        # Equation (3.17)
        session_embedding_all = torch.cat([sess_emb_i, sess_emb_kg], 1)
        item_embeddings_all = torch.cat([item_embeddings_i, item_embeddings_kg], 1)
        scores_item = torch.mm(session_embedding_all, torch.transpose(item_embeddings_all, 1, 0))
        loss_item = self.loss_function(scores_item, tar)
   
        # compute probability of items to be positive examples
        pos_prob_I = self.example_predicting(item_embeddings_i, sess_emb_i)
        pos_prob_S = self.example_predicting(self.embedding, sess_emb_s)

        # choose top-10 items as positive samples and randomly choose 10 items as negative and get their embedding
        pos_emb_I, neg_emb_I, pos_emb_S, neg_emb_S = self.topk_func_random(pos_prob_I,pos_prob_S, item_embeddings_i, self.embedding)

        last_item = torch.squeeze(reversed_sess_item[:, 0])
        last_item = last_item - 1
        last = item_embeddings_i.index_select(0, last_item)
        con_loss = self.SSL_topk(last, sess_emb_i, pos_emb_I, neg_emb_I)
        last = self.embedding[last_item]
        con_loss += self.SSL_topk(last, sess_emb_s, pos_emb_S, neg_emb_S)
        
        reg_kg = 0
        return self.beta * con_loss, loss_item, scores_item[:,:self.n_item], reg_kg

    def test_loss(self, sessionID, session_item, session_len, D, A, reversed_sess_item, mask, epoch, tar, diff_mask, kg):
        self.kg_edge_weight = None

        item_embeddings_i = self.HyperGraph(self.adjacency, self.embedding)
        item_embeddings_kg = self.calc_kg_emb(kg)

        sess_emb_i = self.generate_sess_emb(item_embeddings_i, session_item, session_len, reversed_sess_item, mask)
        sess_emb_kg = self.generate_sess_emb(item_embeddings_kg, session_item, session_len,reversed_sess_item, mask)
        
        sess_emb_i = self.w_k * F.normalize(sess_emb_i, dim=-1, p=2)
        sess_emb_kg = self.w_k * F.normalize(sess_emb_kg, dim=-1, p=2)
        item_embeddings_i = F.normalize(item_embeddings_i, dim=-1, p=2)
        item_embeddings_kg = F.normalize(item_embeddings_kg, dim=-1, p=2)
        session_embedding_all = torch.cat([sess_emb_i, sess_emb_kg], 1)
        item_embeddings_all = torch.cat([item_embeddings_i, item_embeddings_kg], 1)
        scores_item = torch.mm(session_embedding_all, torch.transpose(item_embeddings_all, 1, 0))
        
        loss_item = self.loss_function(scores_item, tar)
        con_loss = 0
        reg_kg = 0
        
        return self.beta * con_loss, loss_item, scores_item[:,:self.n_item], reg_kg
    
    def forward(self, mode, *input):
        if mode == 'train_kg':
            return self.calc_kg_loss(*input)
        if mode == 'train':
            return self.train_loss(*input)
        if mode == 'test':
            return self.test_loss(*input)


def forward(model, i, sessionID, data, kg, epoch, mode):
    tar, session_len, session_item, reversed_sess_item, mask, diff_mask = data.get_slice(i)
    sessionID = trans_to_cuda(torch.Tensor(sessionID).long())
    diff_mask = trans_to_cuda(torch.Tensor(diff_mask).long())
    A_hat, D_hat = data.get_overlap(session_item)
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    A_hat = trans_to_cuda(torch.Tensor(A_hat))
    D_hat = trans_to_cuda(torch.Tensor(D_hat))
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    con_loss, loss_item, scores_item, reg_kg = model(mode, sessionID, session_item, session_len, D_hat, A_hat, reversed_sess_item, mask, epoch,tar, diff_mask, kg)
    return tar, scores_item, con_loss, loss_item, reg_kg


@jit(nopython=True)
def find_k_largest(K, candidates):
    n_candidates = []
    for iid, score in enumerate(candidates[:K]):
        n_candidates.append((score, iid))
    heapq.heapify(n_candidates)
    for iid, score in enumerate(candidates[K:]):
        if score > n_candidates[0][0]:
            heapq.heapreplace(n_candidates, (score, iid + K))
    n_candidates.sort(key=lambda d: d[0], reverse=True)
    ids = [item[1] for item in n_candidates]
    return ids

def train_test(model, train_data, test_data, kg, epoch, drop_rate):
    print('start training: ', datetime.datetime.now())
    total_loss = 0.0
    slices, sessionID = train_data.generate_batch(model.batch_size)
    
    time1 = time()
    kg_total_loss = 0
    n_kg_batch = train_data.n_kg_data // train_data.kg_batch_size + 1

    dropout_rate = drop_rate

    sub_kg_adjM, _ = train_data._get_kg_adj_list(is_subgraph = False, dropout_rate = dropout_rate)
    sub_kg_lap = sum(train_data._get_kg_lap_list(is_subgraph = False, subgraph_adj = sub_kg_adjM))
    sub_kg = dgl.DGLGraph(sub_kg_lap)
    sub_kg = dgl.remove_self_loop(sub_kg)
    sub_kg = dgl.add_self_loop(sub_kg)
    
    sub_kg = sub_kg.to('cuda')
    loss, kge_loss = 0., 0.

    #開始訓練知識圖譜
    for idx in tqdm(range(n_kg_batch)):
        model.zero_grad()
        model.train()
        kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = train_data.generate_kg_batch(train_data.kg_dict)
        kg_batch_head = trans_to_cuda(kg_batch_head)
        kg_batch_relation = trans_to_cuda(kg_batch_relation)
        kg_batch_pos_tail = trans_to_cuda(kg_batch_pos_tail)
        kg_batch_neg_tail = trans_to_cuda(kg_batch_neg_tail)
        kge_loss = model("train_kg", sub_kg, kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail)

        kge_loss.backward()
        model.optimizer.step()
        kg_total_loss += kge_loss.item()
    print('KG Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_kg_batch, time() - time1, kg_total_loss / n_kg_batch))
    
    #i會是一個list，內存當前batch應該取出哪幾個session(index)，例：[0,1,2,3,4,5]，由於數據集本身已被打亂，所以取出的index都會照順序，而不是隨機亂跳
    for i in tqdm(range(len(slices))):
        model.zero_grad()
        tar, scores_item, con_loss, loss_item, reg_kg = forward(model, slices[i], sessionID[i], train_data, sub_kg, epoch, mode='train')
        loss = loss_item + con_loss
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()
    print('\tLoss:\t%.3f' % total_loss)

    top_K = [5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
    print('start predicting: ', datetime.datetime.now())

    model.eval()
    slices, sessionID = test_data.generate_batch(model.batch_size)
    for i in range(len(slices)):
        tar,scores_item, con_loss, loss_item, reg_kg = forward(model, slices[i], sessionID[i], test_data, kg, epoch, mode='test')
        scores = trans_to_cpu(scores_item).detach().numpy()
        index = []
        for idd in range(model.batch_size):
            index.append(find_k_largest(20, scores[idd]))
        index = np.array(index)
        tar = trans_to_cpu(tar).detach().numpy()
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
    return metrics, total_loss
