import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from operator import itemgetter
import random
import pandas as pd
import time
import collections
import torch

# Hypergraph
def data_masks(all_sessions, n_node):
    item_dict = dict()
    inter_mat = list()

    indptr, indices, data = [], [], []
    indptr.append(0)
    n_session = len(all_sessions)
    for j in range(n_session):
        session = np.unique(all_sessions[j])
        length = len(session)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            inter_mat.append([j, session[i]])
            if session[i] not in item_dict.keys():
                item_dict[session[i]] = []
            item_dict[session[i]].append(j)
            indices.append(session[i]-1)
            data.append(1)
    matrix = csr_matrix((data, indices, indptr), shape=(n_session, n_node))
    return matrix, n_session, item_dict, np.array(inter_mat)

class Data():
    def __init__(self, dataset, data, all_train, opt, shuffle=False, n_item=None, n_node=None, KG=False):
        self.raw = np.asarray(data[0], dtype="object")
        H_T, self.n_session, self.item_dict, self.cf_data = data_masks(self.raw, n_node)
        BH_T = H_T.T.multiply(1.0/H_T.sum(axis=1).reshape(1, -1))
        BH_T = BH_T.T
        H = H_T.T
        DH = H.T.multiply(1.0/H.sum(axis=1).reshape(1, -1))
        DH = DH.T
        DHBH_T = np.dot(DH,BH_T)
        self.adjacency = DHBH_T.tocoo()
        self.n_node = n_node
        self.n_items = n_item
        self.targets = np.asarray(data[1])
        self.length = len(self.raw)
        self.shuffle = shuffle
        self.adj_type = opt.adj_type
        self.exist_items = self.item_dict.keys()
        self.adj_list= self._get_cf_adj_list()
        self.lap_list = self._get_lap_list()

        if KG:
            self.kg_batch_size = opt.kg_batch_size
            self.cl_batch_size = opt.batch_size_cl
            kg_data = self.load_kg('../datasets/'+dataset+'/kg.txt')
            print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()), '-- kg data load --')
            self.construct_data(kg_data)
            # generate the triples dictionary, key is 'head', value is '(tail, relation)'
            self.all_kg_dict = self._get_all_kg_dict()
            self.kg_adj_list, self.adj_r_list = self._get_kg_adj_list()
            self.kg_lap_list = self._get_kg_lap_list()

    def get_overlap(self, sessions):
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            seq_a = set(sessions[i])
            seq_a.discard(0)
            for j in range(i+1, len(sessions)):
                seq_b = set(sessions[j])
                seq_b.discard(0)
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a | seq_b
                matrix[i][j] = float(len(overlap))/float(len(ab_set))
                matrix[j][i] = matrix[i][j]
        # matrix = self.dropout(matrix, 0.2)
        matrix = matrix + np.diag([1.0]*len(sessions))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0/degree)
        return matrix, degree

    def generate_batch(self, batch_size):
        shuffled_arg = np.arange(self.length)
        if self.shuffle:
            np.random.shuffle(shuffled_arg)
            self.raw = self.raw[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices, shuffled_arg[slices]

    def get_slice(self, index):
        items, num_node = [], []
        inp = self.raw[index]
        # Obtain the length of each session in the current batch
        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = np.max(num_node)
        session_len = []
        reversed_sess_item = []
        mask = []
        
        for session in inp:
            #假設一個session有[4,45,214,74,4]，下面這行會得到[0,1,2,3,4]
            nonzero_elems = np.nonzero(session)[0]
            # item_set.update(set([t-1 for t in session]))
            session_len.append([len(nonzero_elems)])

            #Pad all sessions to the same length, which is the length of the longest session in the current batch
            items.append(session + (max_n_node - len(nonzero_elems)) * [0])
            mask.append([1]*len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
            
            # Reverse the order of items within the session
            reversed_sess_item.append(list(reversed(session)) + (max_n_node - len(nonzero_elems)) * [0])

        diff_mask = np.ones(shape=[100, self.n_node]) * (1/(self.n_node - 1))

        for count, value in enumerate(self.targets[index]-1):
            diff_mask[count][value] = 1
        return self.targets[index]-1, session_len,items, reversed_sess_item, mask, diff_mask
    
    def load_kg(self, filename):
        kg_data = pd.read_csv(filename, sep=' ', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()
        return kg_data

    def construct_data(self, kg_data):
        # plus inverse kg data
        n_relations = max(kg_data['r']) + 1
        reverse_kg_data = kg_data.copy()
        reverse_kg_data = reverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        # print(reverse_kg_data)  # t r h [2557746 rows x 3 columns]
        reverse_kg_data['r'] += n_relations
        # print(reverse_kg_data)  # t r h [2557746 rows x 3 columns]
        self.kg_data = pd.concat([kg_data, reverse_kg_data], axis=0, ignore_index=True, sort=False)
        # print(kg_data)  # h r t [5115492 rows x 3 columns]

        # re-map user id
        self.n_relations = max(self.kg_data['r']) + 1
        self.n_entities = max(max(kg_data['h']), max(kg_data['t']))
        self.n_kg_data = len(self.kg_data)

        # construct kg dict
        self.kg_dict = collections.defaultdict(list)
        self.relation_dict = collections.defaultdict(list)
        # self.train_relation_dict = collections.defaultdict(list)
        for row in self.kg_data.iterrows():
            h, r, t = row[1]
            self.kg_dict[h].append((t, r))
            self.relation_dict[r].append((h, t))

    def sample_pos_triples_for_h(self, kg_dict, head, n_sample_pos_triples):
        pos_triples = kg_dict[head]
        n_pos_triples = len(pos_triples)

        sample_relations, sample_pos_tails = [], []
        while True:
            if len(sample_relations) == n_sample_pos_triples:
                break

            pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
            tail = pos_triples[pos_triple_idx][0]
            relation = pos_triples[pos_triple_idx][1]

            if relation not in sample_relations and tail not in sample_pos_tails:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails

    def sample_neg_triples_for_h(self, kg_dict, head, relation, n_sample_neg_triples):
        pos_triples = kg_dict[head]

        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == n_sample_neg_triples:
                break
            tail = np.random.randint(low=0, high=self.n_entities, size=1)[0]
            if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
                sample_neg_tails.append(tail)
        return sample_neg_tails


    def generate_kg_batch(self, kg_dict):
        exist_heads = kg_dict.keys()
        if self.kg_batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, self.kg_batch_size)
        else:
            batch_head = [random.choice(exist_heads) for _ in range(self.kg_batch_size)]

        batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
        for h in batch_head:
            relation, pos_tail = self.sample_pos_triples_for_h(kg_dict, h, 1)
            batch_relation += relation
            batch_pos_tail += pos_tail

            neg_tail = self.sample_neg_triples_for_h(kg_dict, h, relation[0], 1)
            batch_neg_tail += neg_tail

        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)
        return batch_head, batch_relation, batch_pos_tail, batch_neg_tail
    
    def _get_kg_adj_list(self, is_subgraph = False, dropout_rate = None):
        adj_mat_list = []
        adj_r_list = []
        # Build an adjacency matrix for each relation
        def _np_mat2sp_adj(np_mat):
            n_all = self.n_entities
            # single-direction
            a_rows = np_mat[:, 0]-1
            a_cols = np_mat[:, 1]-1
            if is_subgraph is True:
                subgraph_idx = np.arange(len(a_rows))
                subgraph_id = np.random.choice(subgraph_idx, size = int(dropout_rate * len(a_rows)), replace = False)
                #print(subgraph_id[:10])
                a_rows = a_rows[subgraph_id]
                a_cols = a_cols[subgraph_id]
            a_vals = [1.] * len(a_rows)

            a_adj = coo_matrix((a_vals, (a_rows, a_cols)), shape=(n_all, n_all))

            return a_adj
        
        # relation_dict records which (head, tail) pairs each relation has
        for r_id in self.relation_dict.keys():
            #print(r_id)
            K = _np_mat2sp_adj(np.array(self.relation_dict[r_id]))
            adj_mat_list.append(K)
            adj_r_list.append(r_id)
        #print(adj_r_list)
        return adj_mat_list, adj_r_list
    
    def _bi_norm_lap(self, adj):
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = diags(d_inv_sqrt)

        bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(self, adj):
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()
    
    def _get_kg_lap_list(self, is_subgraph = False, subgraph_adj = None):
        if is_subgraph is True:
            adj_list = subgraph_adj
        else:
            adj_list = self.kg_adj_list
        if self.adj_type == 'bi':
            lap_list = [self._bi_norm_lap(adj) for adj in adj_list]
        else:
            lap_list = [self._si_norm_lap(adj) for adj in adj_list]
        return lap_list
    
    # Store which entities each head is associated with and what the relation is, for example: {head: (tail, relation)}
    def _get_all_kg_dict(self):
        # When encountering a non-existent key, it generates a default value, which is an empty list
        all_kg_dict = collections.defaultdict(list)
        
        for relation in self.relation_dict.keys():
            for head, tail in self.relation_dict[relation]:
                all_kg_dict[head].append((tail, relation))
        return all_kg_dict
    
    def _generate_train_cl_batch(self):
        if self.cl_batch_size <= len(self.exist_items):
            items = random.sample(self.exist_items, self.cl_batch_size)
        else:
            items_list = list(self.exist_items)
            items = [random.choice(items_list) for _ in range(self.cl_batch_size)]
        return items

    def generate_train_cl_batch(self):
        items = self._generate_train_cl_batch()
        batch_data = {}
        batch_data['items'] = items
        return batch_data
    
    def _get_cf_adj_list(self, is_subgraph = False, dropout_rate = None):
        def _np_mat2sp_adj(np_mat, row_pre, col_pre):
            n_all = self.n_session + self.n_items
            # single-direction
            a_rows = np_mat[:, 0] + row_pre
            a_cols = np_mat[:, 1] + col_pre -1
            if is_subgraph is True:
                subgraph_idx = np.arange(len(a_rows))
                subgraph_id = np.random.choice(subgraph_idx, size = int(dropout_rate * len(a_rows)), replace = False)
                a_rows = a_rows[subgraph_id]
                a_cols = a_cols[subgraph_id]
                
            vals = [1.] * len(a_rows) * 2
            rows = np.concatenate((a_rows, a_cols))
            cols = np.concatenate((a_cols, a_rows))
            adj = coo_matrix((vals, (rows, cols)), shape=(n_all, n_all))
            return adj
        R = _np_mat2sp_adj(self.cf_data, row_pre=0, col_pre=self.n_session)
        return R
    
    def _get_lap_list(self, is_subgraph = False, subgraph_adj = None):
        if is_subgraph is True:
            adj = subgraph_adj
        else:
            adj = self.adj_list
        if self.adj_type == 'bi':
            lap_list = self._bi_norm_lap(adj)
        else:
            lap_list = self._si_norm_lap(adj)
        return lap_list
