# coding: utf-8
# @email: georgeguo.gzq.cn@gmail.com
r"""
LGMRec
################################################
Reference:
    https://github.com/georgeguo-cn/LGMRec
    AAAI'2024: [LGMRec: Local and Global Graph Learning for Multimodal Recommendation]
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender

class YunJian_v53(GeneralRecommender):
    def __init__(self, config, dataset):
        super(YunJian_v53, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.cf_model = config['cf_model']
        self.n_mm_layer = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.keep_rate = config['keep_rate']
        self.reg_weight = config['reg_weight']

        self.n_nodes = self.n_users + self.n_items

        self.mge_weight = config['mge_weight']
        self.relation_distillation_func = config['relation_distillation_func']
        self.behavior_distillation_weight = config['behavior_distillation_weight']
        self.visual_distillation_weight = config['visual_distillation_weight']
        self.textual_distillation_weight = config['textual_distillation_weight']
        self.behavior_graph_dropout_threshold = config['behavior_graph_dropout_threshold']
        self.image_knn_k = config['image_knn_k']
        self.text_knn_k = config['text_knn_k']
        self.behavior_knn_k = config['behavior_knn_k']

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.adj = self.scipy_matrix_to_sparse_tenser(self.interaction_matrix, torch.Size((self.n_users, self.n_items)))
        self.num_inters, self.norm_adj = self.get_norm_adj_mat()
        self.num_inters = torch.FloatTensor(1.0 / (self.num_inters + 1e-7)).to(self.device)
        
        # init user and item ID embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)


        # load item modal features and define hyperedges embeddings
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)
            self.item_image_trs = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.v_feat.shape[1], self.feat_embed_dim)))
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)
            self.item_text_trs = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.t_feat.shape[1], self.feat_embed_dim)))


        if self.behavior_distillation_weight > 0:
            behavior_i2i_dense_adj = torch.sparse.mm(self.adj.transpose(1,0), self.adj).to_dense()
            torch.diagonal(behavior_i2i_dense_adj).fill_(0)
            self.b_topk_values, self.b_topk_indices = torch.topk(behavior_i2i_dense_adj, k=self.behavior_knn_k,dim=1, largest=True, sorted=True)
            self.b_zero_mask = torch.where(self.b_topk_values > self.behavior_graph_dropout_threshold, torch.tensor(1.0).to(self.device), torch.tensor(0.0).to(self.device))
            i2i_b_relation_masked = self.b_topk_values.masked_fill(~self.b_zero_mask.bool(), torch.tensor(float('-inf')).to(self.device))
            self.i2i_b_distribution = F.softmax(i2i_b_relation_masked, dim=-1)
            self.i2i_b_distribution = torch.where(torch.isnan(self.i2i_b_distribution), torch.tensor(0.0).to(self.device), self.i2i_b_distribution)

        if self.visual_distillation_weight > 0:
            i_v_feat = F.normalize(self.v_feat, dim=1)
            self.i2i_v_relation = torch.matmul(i_v_feat, i_v_feat.T)
            torch.diagonal(self.i2i_v_relation).fill_(0)
            self.v_topk_values, self.v_topk_indices = torch.topk(self.i2i_v_relation, k=self.image_knn_k, dim=1, largest=True, sorted=True)
            self.i2i_v_distribution = F.softmax(self.v_topk_values, dim=-1)
        
        if self.textual_distillation_weight > 0:
            i_t_feat = F.normalize(self.t_feat, dim=1)
            self.i2i_t_relation = torch.matmul(i_t_feat, i_t_feat.T)
            torch.diagonal(self.i2i_t_relation).fill_(0)
            self.t_topk_values, self.t_topk_indices = torch.topk(self.i2i_t_relation, k=self.text_knn_k, dim=1, largest=True, sorted=True)
            self.i2i_t_distribution = F.softmax(self.t_topk_values, dim=-1)
        

    def scipy_matrix_to_sparse_tenser(self, matrix, shape):
        row = matrix.row
        col = matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(matrix.data)
        return torch.sparse.FloatTensor(i, data, shape).to(self.device)
    
    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        return sumArr, self.scipy_matrix_to_sparse_tenser(L, torch.Size((self.n_nodes, self.n_nodes)))
    
    # collaborative graph embedding
    def cge(self):
        if self.cf_model == 'mf':
            cge_embs = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        if self.cf_model == 'lightgcn':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            cge_embs = [ego_embeddings]
            for _ in range(self.n_ui_layers):
                ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
                cge_embs += [ego_embeddings]
            cge_embs = torch.stack(cge_embs, dim=1)
            cge_embs = cge_embs.mean(dim=1, keepdim=False)
        return cge_embs
    
    # modality graph embedding
    def mge(self, str='v'):
        if str == 'v':
            item_feats = torch.mm(self.image_embedding.weight, self.item_image_trs)
        elif str == 't':
            item_feats = torch.mm(self.text_embedding.weight, self.item_text_trs)
        user_feats = torch.sparse.mm(self.adj, item_feats) * self.num_inters[:self.n_users]
        # user_feats = self.user_embedding.weight
        mge_feats = torch.concat([user_feats, item_feats], dim=0)
        for _ in range(self.n_mm_layer):
            mge_feats = torch.sparse.mm(self.norm_adj, mge_feats)
        return mge_feats
    
    def forward(self):
        # CGE: collaborative graph embedding
        cge_embs = self.cge()
        
        if self.v_feat is not None and self.t_feat is not None:
            # MGE: modal graph embedding
            v_feats = self.mge('v')
            t_feats = self.mge('t')
            # local embeddings = collaborative-related embedding + modality-related embedding
            norm_v_feats, norm_t_feats = F.normalize(v_feats), F.normalize(t_feats)
            lge_embs = cge_embs + self.mge_weight * (norm_v_feats + norm_t_feats)
            all_embs = lge_embs
        else:
            all_embs = cge_embs

        u_embs, i_embs = torch.split(all_embs, [self.n_users, self.n_items], dim=0)
        uv_embs, iv_embs = torch.split(norm_v_feats, [self.n_users, self.n_items], dim=0)
        ut_embs, it_embs = torch.split(norm_t_feats, [self.n_users, self.n_items], dim=0)

        return u_embs, i_embs, uv_embs, iv_embs, ut_embs, it_embs
        
    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return bpr_loss
    
    def ssl_triple_loss(self, emb1, emb2, all_emb):
        norm_emb1 = F.normalize(emb1)
        norm_emb2 = F.normalize(emb2)
        norm_all_emb = F.normalize(all_emb)
        pos_score = torch.exp(torch.mul(norm_emb1, norm_emb2).sum(dim=1) / self.tau)
        ttl_score = torch.exp(torch.matmul(norm_emb1, norm_all_emb.T) / self.tau).sum(dim=1)
        ssl_loss = -torch.log(pos_score / ttl_score).sum()
        return ssl_loss
    
    def reg_loss(self, *embs):
        reg_loss = 0
        for emb in embs:
            reg_loss += torch.norm(emb, p=2)
        reg_loss /= embs[-1].shape[0]
        return reg_loss

    def calculate_loss(self, interaction):
        ua_embeddings, ia_embeddings, uv_embeddings, iv_embeddings, ut_embeddings, it_embeddings = self.forward()

        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_bpr_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        batch_reg_loss = self.reg_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        loss = batch_bpr_loss + self.reg_weight * batch_reg_loss

        if self.behavior_distillation_weight > 0:
            ia_embeddings = F.normalize(ia_embeddings, dim=1)
            ia_topk_embeddigns = ia_embeddings[self.b_topk_indices.view(-1)].view(self.b_topk_indices.shape[0], self.behavior_knn_k, -1) # [item_num, k, dim]
            i2i_a_topk_similarity = torch.einsum('nd,nkd->nk', ia_embeddings, ia_topk_embeddigns) # [item_num, topk]

            i2i_a_similarity_masked = i2i_a_topk_similarity.masked_fill(~self.b_zero_mask.bool(), torch.tensor(float('-inf')).to(self.device))
            i2i_a_probability = F.softmax(i2i_a_similarity_masked, dim=-1)
            i2i_a_probability = torch.where(torch.isnan(i2i_a_probability), torch.tensor(1e-9).to(self.device), i2i_a_probability)

            if self.relation_distillation_func == "CE":
                behavior_distillation_loss = -1 * self.i2i_b_distribution * torch.log(i2i_a_probability + 1e-9)
            elif self.relation_distillation_func == "KL":
                behavior_distillation_loss = self.i2i_b_distribution * torch.log((self.i2i_b_distribution + 1e-9) / (i2i_a_probability + 1e-9))
            else:
                raise NotImplementedError

            loss += self.behavior_distillation_weight * behavior_distillation_loss.sum(dim=-1).mean()

        if self.visual_distillation_weight > 0:
            iv_topk_embeddings = iv_embeddings[self.v_topk_indices.view(-1)].view(self.v_topk_indices.shape[0], self.image_knn_k, -1) # [item_num, k, dim]
            i2i_v_topk_similarity = torch.einsum('nd,nkd->nk', iv_embeddings, iv_topk_embeddings) # [item_num, topk]
            i2i_v_probability = F.softmax(i2i_v_topk_similarity, dim=-1)

            if self.relation_distillation_func == "CE":
                visual_distillation_loss = -1 * self.i2i_v_distribution * torch.log(i2i_v_probability)
            elif self.relation_distillation_func == "KL":
                visual_distillation_loss = self.i2i_v_distribution * torch.log(self.i2i_v_distribution / i2i_v_probability)
            else:
                raise NotImplementedError
            loss += self.visual_distillation_weight * visual_distillation_loss.sum(dim=-1).mean()

        if self.textual_distillation_weight > 0:
            it_topk_embeddings = it_embeddings[self.t_topk_indices.view(-1)].view(self.t_topk_indices.shape[0], self.text_knn_k, -1) # [item_num, k, dim]
            i2i_t_topk_similarity = torch.einsum('nd,nkd->nk', it_embeddings, it_topk_embeddings) # [item_num, topk]
            i2i_t_probability = F.softmax(i2i_t_topk_similarity, dim=-1)

            if self.relation_distillation_func == "CE":
                textual_distillation_loss = -1 * self.i2i_t_distribution * torch.log(i2i_t_probability)
            elif self.relation_distillation_func == "KL":
                textual_distillation_loss = self.i2i_t_distribution * torch.log(self.i2i_t_distribution / i2i_t_probability)
            else:
                raise NotImplementedError
            loss += self.textual_distillation_weight * textual_distillation_loss.sum(dim=-1).mean()

        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embs, item_embs, _, _, _, _ = self.forward()
        scores = torch.matmul(user_embs[user], item_embs.T)
        return scores

