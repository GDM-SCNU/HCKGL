import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from rec.model.abstract_recommender import KnowledgeRecommender
from rec.model.init import xavier_normal_initialization
from rec.model.loss import BPRLoss, EmbLoss
from rec.utils import InputType

from utils.euclidean import givens_rotations, givens_reflection
from utils.hyperbolic import mobius_add, expmap0, project,score,hyp_distance_multi_c


class Aggregator(nn.Module):

    def __init__(self, input_dim, output_dim, dropout, aggregator_type = 'gcn'):
        super(Aggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)

        if self.aggregator_type == 'gcn':
            self.W = nn.Linear(self.input_dim, self.output_dim)
        elif self.aggregator_type == 'graphsage':
            self.W = nn.Linear(self.input_dim * 2, self.output_dim)
        elif self.aggregator_type == 'bi':
            self.W1 = nn.Linear(self.input_dim, self.output_dim)
            self.W2 = nn.Linear(self.input_dim, self.output_dim)
        else:
            raise NotImplementedError

        self.activation = nn.LeakyReLU()

    def forward(self, norm_matrix, ego_embeddings):
        side_embeddings = torch.sparse.mm(norm_matrix, ego_embeddings)

        if self.aggregator_type == 'gcn':
            ego_embeddings = self.activation(self.W(ego_embeddings + side_embeddings))
        elif self.aggregator_type == 'graphsage':
            ego_embeddings = self.activation(self.W(torch.cat([ego_embeddings, side_embeddings], dim=1)))
        elif self.aggregator_type == 'bi':
            add_embeddings = ego_embeddings + side_embeddings
            sum_embeddings = self.activation(self.W1(add_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = self.activation(self.W2(bi_embeddings))
            ego_embeddings = bi_embeddings + sum_embeddings
        else:
            raise NotImplementedError

        ego_embeddings = self.message_dropout(ego_embeddings)

        return ego_embeddings

class HCKGL(KnowledgeRecommender):

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(HCKGL, self).__init__(config, dataset)

        # load dataset info
        self.ckg = dataset.ckg_graph(form='dgl', value_field='relation_id')
        self.all_hs = torch.LongTensor(dataset.ckg_graph(form='coo', value_field='relation_id').row).to(self.device)
        self.all_ts = torch.LongTensor(dataset.ckg_graph(form='coo', value_field='relation_id').col).to(self.device)
        self.all_rs = torch.LongTensor(dataset.ckg_graph(form='coo', value_field='relation_id').data).to(self.device)
        self.matrix_size = torch.Size([self.n_users + self.n_entities, self.n_users + self.n_entities])

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.kg_embedding_size = config['kg_embedding_size']
        self.layers = [self.embedding_size] + config['layers']
        self.aggregator_type = config['aggregator_type']
        self.mess_dropout = config['mess_dropout']
        self.reg_weight = config['reg_weight']

        # generate intermediate data
        self.A_in = self.init_graph()  # init the attention matrix by the structure of ckg
        self.A_in_1 = self.A_in
        self.A_in_2 = self.A_in
        affine = True
        self.projection_head = torch.nn.ModuleList()
        self.inner_size = self.layers[-1] * 2
        # print("==================inner size:===================", inner_size)
        self.projection_head.append(torch.nn.Linear(self.inner_size, self.inner_size * 4, bias=False))
       
        self.projection_head.append(torch.nn.BatchNorm1d(self.inner_size * 4, eps=1e-12, affine=affine))
        self.projection_head.append(torch.nn.Linear(self.inner_size * 4, self.inner_size, bias=False))
        self.projection_head.append(torch.nn.BatchNorm1d(self.inner_size, eps=1e-12, affine=affine))
        self.mode = 0


        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        # self.relation_embedding = nn.Embedding(self.n_relations, self.kg_embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size)
        self.trans_w = nn.Embedding(self.n_relations, self.embedding_size * self.kg_embedding_size)
        self.aggregator_layers = nn.ModuleList()
        for idx, (input_dim, output_dim) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.aggregator_layers.append(Aggregator(input_dim, output_dim, self.mess_dropout, self.aggregator_type))

        self.tanh = nn.Tanh()
        self.mf_loss = BPRLoss() #
        self.reg_loss = EmbLoss() #
        self.ce_loss = nn.CrossEntropyLoss()
        self.restore_user_e = None
        self.restore_entity_e = None

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_entity_e']

        self.init_size=1e-3
        self.bh = nn.Embedding(self.n_entities, 1)
        self.bh.weight.data = torch.zeros((self.n_entities, 1))
        self.bt = nn.Embedding(self.n_entities, 1)
        self.bt.weight.data = torch.zeros((self.n_entities, 1))
        self.rel_diag = nn.Embedding(self.n_relations, 2 * self.embedding_size) 
        self.rel_diag.weight.data = 2 * torch.rand((self.n_relations, 2 * self.embedding_size)) - 1.0
        self.context_vec = nn.Embedding(self.n_relations, self.embedding_size).to(self.device)#
        self.context_vec.weight.data = self.init_size * torch.randn((self.n_relations, self.embedding_size))
        self.act = nn.Softmax(dim=1)
        self.scale = torch.Tensor([1. / np.sqrt(self.embedding_size)]).double().to(self.device)
        self.c = nn.Parameter(torch.ones((self.n_relations, 1), dtype=torch.float32), requires_grad=True)#################################################


    def init_graph(self):  #

        import dgl   #
        adj_list = []   #
        for rel_type in range(1, self.n_relations, 1):
            edge_idxs = self.ckg.filter_edges(lambda edge: edge.data['relation_id'] == rel_type) #
            #
            sub_graph = dgl.edge_subgraph(self.ckg, edge_idxs, relabel_nodes=False). \
                adjacency_matrix(transpose=False, scipy_fmt='coo').astype('float')    
             #
            rowsum = np.array(sub_graph.sum(1))  
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)    
            norm_adj = d_mat_inv.dot(sub_graph).tocoo()
            adj_list.append(norm_adj)  #
            
        #
        final_adj_matrix = sum(adj_list).tocoo()
        indices = torch.LongTensor([final_adj_matrix.row, final_adj_matrix.col])
        values = torch.FloatTensor(final_adj_matrix.data)
        adj_matrix_tensor = torch.sparse.FloatTensor(indices, values, self.matrix_size)
        return adj_matrix_tensor.to(self.device)
        
     #
    def _get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        entity_embeddings = self.entity_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, entity_embeddings], dim=0)#
        return ego_embeddings
    
    
    def forward(self):
        ego_embeddings = self._get_ego_embeddings()
        embeddings_list = [ego_embeddings]
        #
        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in, ego_embeddings)   #
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings) #
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)   #
        user_all_embeddings, entity_all_embeddings = torch.split(kgat_all_embeddings, [self.n_users, self.n_entities])
        return user_all_embeddings, entity_all_embeddings

    def forward_1(self):
        ego_embeddings = self._get_ego_embeddings()
        embeddings_list = [ego_embeddings]
        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in_1, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        user_all_embeddings, entity_all_embeddings = torch.split(kgat_all_embeddings, [self.n_users, self.n_entities])
        return user_all_embeddings, entity_all_embeddings

    def forward_2(self):
        ego_embeddings = self._get_ego_embeddings()
        
        # 
        x = 0.001 + torch.zeros(ego_embeddings.shape[0], ego_embeddings.shape[1], dtype=torch.float32, device=ego_embeddings.device) # add guassian noise.
        noise = torch.normal(mean=torch.tensor([0.0]).to(ego_embeddings.device),std=x).to(ego_embeddings.device) # add guassian noise.
        ego_embeddings=ego_embeddings+noise

        embeddings_list = [ego_embeddings]
        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in_2, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        user_all_embeddings, entity_all_embeddings = torch.split(kgat_all_embeddings, [self.n_users, self.n_entities])
        return user_all_embeddings, entity_all_embeddings
    
   
        
    #
    def mask_correlated_samples(self, batch_size): #
        N = 2 * batch_size #
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)  #
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask


    def _get_kg_embedding(self, h, r, pos_t, neg_t):
        h_e = self.entity_embedding(h).unsqueeze(1)
        pos_t_e = self.entity_embedding(pos_t).unsqueeze(1)
        neg_t_e = self.entity_embedding(neg_t).unsqueeze(1)
        r_e = self.relation_embedding(r)
        r_trans_w = self.trans_w(r).view(r.size(0), self.embedding_size, self.kg_embedding_size)

        h_e = torch.bmm(h_e, r_trans_w).squeeze(1)
        pos_t_e = torch.bmm(pos_t_e, r_trans_w).squeeze(1)
        neg_t_e = torch.bmm(neg_t_e, r_trans_w).squeeze(1)

        return h_e, r_e, pos_t_e, neg_t_e
        
     #
    def cts_loss(self, z_i, z_j, temp, batch_size): #
        
        N = 2 * batch_size
    
        z = torch.cat((z_i, z_j), dim=0)   #
    
        sim = torch.mm(z, z.T) / temp   # 2B * 2B
        
        #
        sim_i_j = torch.diag(sim, batch_size)    #B*1 
        sim_j_i = torch.diag(sim, -batch_size)   #B*1
    
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1) 

        mask = self.mask_correlated_samples(batch_size)   

        negative_samples = sim[mask].reshape(N, -1)
    
        labels = torch.zeros(N).to(positive_samples.device).long() 
        logits = torch.cat((positive_samples, negative_samples), dim=1)  # N * C
        loss = self.ce_loss(logits, labels) 
        return loss
        
    #
    def projection_head_map(self, state, mode):
        for i, l in enumerate(self.projection_head): # 0: Linear 1: BN (relu)  2: Linear 3:BN (relu)
            if i % 2 != 0:
                if mode == 0:
                    l.train()   # set BN to train mode: use a learned mean and variance.
                else:
                    l.eval()   # set BN to eval mode: use a accumulated mean and variance.
            state = l(state)
            if i % 2 != 0:
                state = F.relu(state)   
        return state
 
    def calculate_loss(self, interaction): 
        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        # get loss for training rs
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]


        user_all_embeddings, entity_all_embeddings = self.forward()
       

        kgat_all_embeddings = torch.cat((user_all_embeddings, entity_all_embeddings), 0)


        user_all_embeddings_1, entity_all_embeddings_1 = self.forward_1()
        user_all_embeddings_2, entity_all_embeddings_2 = self.forward_2()


        user_rand_samples = self.rand_sample(user_all_embeddings_1.shape[0], size=user.shape[0]//8, replace=False)#随机抽取出一定数量的用户embedding。
        entity_rand_samples = self.rand_sample(entity_all_embeddings_1.shape[0], size=user.shape[0], replace=False)
        

        cts_embedding_1 = user_all_embeddings_1[torch.tensor(user_rand_samples)]
        cts_embedding_2 = user_all_embeddings_2[torch.tensor(user_rand_samples)]

        e_cts_embedding_1 = entity_all_embeddings_1[torch.tensor(entity_rand_samples)]
        e_cts_embedding_2 = entity_all_embeddings_2[torch.tensor(entity_rand_samples)]

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = entity_all_embeddings[pos_item]
        neg_embeddings = entity_all_embeddings[neg_item]

        

        cts_embedding_1 =self.projection_head_map(cts_embedding_1, self.mode)  #self.mode投影头一般是一个全连接层，通过对节点embedding进行线性变换来获得低维向量表示
        cts_embedding_2 = self.projection_head_map(cts_embedding_2, 1 - self.mode)
        e_cts_embedding_1 = self.projection_head_map(e_cts_embedding_1, self.mode)
        e_cts_embedding_2 = self.projection_head_map(e_cts_embedding_2, 1 - self.mode)

        u_embeddings = self.projection_head_map(u_embeddings, self.mode)
        pos_embeddings = self.projection_head_map(pos_embeddings, 1 - self.mode)



        self.mode = 1 - self.mode    

       

        cts_loss = self.cts_loss(cts_embedding_1, cts_embedding_2, temp=1.0,
                                                        batch_size=cts_embedding_1.shape[0])
                                                        
        e_cts_loss = self.cts_loss(e_cts_embedding_1, e_cts_embedding_2, temp=1.0,
                                                        batch_size=e_cts_embedding_1.shape[0])

        ui_cts_loss = self.cts_loss(u_embeddings, pos_embeddings, temp=1.0,
                                                        batch_size=u_embeddings.shape[0])

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = entity_all_embeddings[pos_item]
        neg_embeddings = entity_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)  #
        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)  #
#        print("cts_loss:", cts_loss, e_cts_loss, ui_cts_loss)
        loss = mf_loss + self.reg_weight * reg_loss + 0.03 * (cts_loss + e_cts_loss + ui_cts_loss) 

        return loss
  
    def calculate_kg_loss(self, interaction):

        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        # get loss for training kg
        h = interaction[self.HEAD_ENTITY_ID]
        r = interaction[self.RELATION_ID]
        pos_t = interaction[self.TAIL_ENTITY_ID]
        neg_t = interaction[self.NEG_TAIL_ENTITY_ID]


        lhs_e, lhs_biases = self.get_queries(h,r) 

        rhs_e, rhs_biases = self.get_rhs(pos_t,self.mode) 
        pos_score = score((lhs_e, lhs_biases), (rhs_e, rhs_biases),self.mode) 
        positive_score = F.logsigmoid(pos_score)
        

        rhs_e, rhs_biases = self.get_rhs(neg_t,self.mode) 
        neg_score = score((lhs_e, lhs_biases), (rhs_e, rhs_biases), self.mode) 
        negative_score = F.logsigmoid(-neg_score)

        loss = - torch.cat([positive_score, negative_score], dim=0).mean()
        
 
        return loss 

   
    def generate_transHyper_score(self, hs, ts, r): #

        all_embeddings = self._get_ego_embeddings()
        h_e = all_embeddings[hs]
        t_e = all_embeddings[ts]
        r_e = self.relation_embedding.weight[r]
        r_trans_w = self.trans_w.weight[r].view(self.embedding_size, self.kg_embedding_size)

        c = F.softplus(self.c[r]) #
        queries  = project(mobius_add(h_e, r_e, c), c) #

        queries= torch.matmul(queries, r_trans_w)
        t_e = torch.matmul(t_e, r_trans_w)
        kg_score = torch.mul(t_e, self.tanh(queries)).sum(dim=1)

        # print('============score==============',kg_score )

        return kg_score

    def rand_sample(self, high, size=None, replace=True):

        a = np.arange(high)
        sample = np.random.choice(a, size=size, replace=replace)
        return sample

    def update_attentive_A(self):

        kg_score_list, row_list, col_list = [], [], []
        # To reduce the GPU memory consumption, we calculate the scores of KG triples according to the type of relation
        for rel_idx in range(1, self.n_relations, 1): #
            triple_index = torch.where(self.all_rs == rel_idx) #
            kg_score = self.generate_transHyper_score(self.all_hs[triple_index], self.all_ts[triple_index], rel_idx) 
           
            row_list.append(self.all_hs[triple_index])
            col_list.append(self.all_ts[triple_index])
            kg_score_list.append(kg_score)
            
        kg_score = torch.cat(kg_score_list, dim=0)
        row = torch.cat(row_list, dim=0)
        col = torch.cat(col_list, dim=0)
        indices = torch.cat([row, col], dim=0).view(2, -1) #行列
       
        A_in = torch.sparse.FloatTensor(indices, kg_score, self.matrix_size).cpu() #
        A_in = torch.sparse.softmax(A_in, dim=1).to(self.device) #
        
        drop_edge_1 = self.rand_sample(indices.shape[1], size=int(indices.shape[1] * 0.1), replace=False)
        indices_1 = indices.view(-1, 2)[torch.tensor(drop_edge_1)].view(2, -1)
        kg_score_1 = kg_score[torch.tensor(drop_edge_1)]
        
        A_in_1 = torch.sparse.FloatTensor(indices_1, kg_score_1, self.matrix_size).cpu()
        A_in_1 = torch.sparse.softmax(A_in_1, dim=1).to(self.device)
         
        drop_edge_2 = self.rand_sample(indices.shape[1], size=int(indices.shape[1] * 0.1), replace=False)
        indices_2 = indices.view(-1, 2)[torch.tensor(drop_edge_2)].view(2, -1)
        kg_score_2 = kg_score[torch.tensor(drop_edge_2)]
        A_in_2 = torch.sparse.FloatTensor(indices_2, kg_score_2, self.matrix_size).cpu()
        A_in_2 = torch.sparse.softmax(A_in_2, dim=1).to(self.device)
        
        self.A_in = A_in
   
        self.A_in_1 = A_in_1
        self.A_in_2 = A_in_2
        

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, entity_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = entity_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_entity_e is None:
            self.restore_user_e, self.restore_entity_e = self.forward()
        u_embeddings = self.restore_user_e[user]
        i_embeddings = self.restore_entity_e[:self.n_items]

        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))

        return scores.view(-1)
