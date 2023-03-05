from locale import normalize
from turtle import distance
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from torch.nn.init import xavier_normal_

class MSINE(SequentialRecommender):
    def __init__(self, config, dataset, num):
        super(MSINE, self).__init__(config, dataset)

        # load parameters info
        self.D = config['embedding_size']
        self.device = config['device']
        self.loss_type = config['loss_type']
        self.layer_norm_eps = config['layer_norm_eps']
        self.L = config['prototype_size']  # 500 for movie-len dataset
        self.k = config['his_interest_size']
        self.tau = config['tau']
        self.temperature = config['temperature']
        self.Dlr = config['Dlr']
        self.G1lr = config['G1lr']
        self.G2lr = config['G2lr']
        self.gama = config['gama']
        self.epochs = 0
        self.num = num
        self.initializer_range = config['initializer_range']

        self.w1 = self._init_weight((self.D, self.D))
        self.w2 = self._init_weight(self.D)
        self.w3 = self._init_weight((self.D, self.D))
        self.w4 = self._init_weight((self.D, self.D))
        self.gen1 = GENERATOR1(config, dataset)
        self.gen2 = GENERATOR2(config, dataset)
        self.dis = DISCRIMINATOR(config, dataset)
        self.transfer = self.trans_init_weight((self.L, self.L))
        # item embedding
        self.item_embedding = nn.Embedding(self.n_items, self.D, padding_idx=0)
        self.C_embedding = nn.Embedding(self.L, self.D)

        if self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        # parameters initialization
        self._reset_parameters()

    def trans_init_weight(self, shape):
        mat = np.random.normal(0, self.initializer_range, shape)
        mat += np.eye(shape[0])
        return nn.Parameter(torch.FloatTensor(mat)).to(self.device)

    def _init_weight(self, shape):
        mat = np.random.normal(0, self.initializer_range, shape)
        return nn.Parameter(torch.FloatTensor(mat)).to(self.device)

    def _reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.D)
        for name, weight in self.named_parameters():
            if 'transfer' not in name:
                weight.data.uniform_(-stdv, stdv)

    def forward(self, item_seq, isTrain):
        """w1 w2 w3 w4 gen1 gen2 dis item_embedding.weight C_embedding.weight"""
        if isTrain:
            self.epochs += 1
        x_u = self.item_embedding(item_seq)
        x_u = item_seq.gt(0).to(torch.float).unsqueeze(-1) * x_u #to keep no-interaction element be zero
        alpha = self.net(item_seq, x_u)
        general_inten = torch.sum(alpha * x_u, dim=1)
        if isTrain:
            noise1 = 0.05*torch.randn(self.C_embedding.weight.shape).to(self.device)
        else:
            noise1 = 0 
        s_u = torch.matmul(general_inten, (self.C_embedding.weight+noise1).transpose(0,1))
        new_s_u = torch.matmul(F.softmax(s_u/0.6,dim=1), F.softmax(self.transfer/0.6,dim=-1))
        C_u = self.C_embedding(new_s_u.argsort(1)[:, -self.k:])

        # intention assignment
        # use matrix multiplication instead of cos()
        curr_inten = self.gen1(C_u, isTrain)
        w3_x_u_norm = F.normalize(x_u.matmul(self.w3), p=2, dim=2)
        P_k_t = torch.bmm(w3_x_u_norm, C_u.transpose(1, 2))
        P_k_t_b = F.softmax(P_k_t, dim=2).transpose(1,2)

        # interest embedding generation
        x_u_re = x_u.unsqueeze(1).repeat(1, self.k, 1, 1)
        mul_p_re = P_k_t_b.unsqueeze(3)
        delta_k = x_u_re.mul(mul_p_re).sum(2)

        # aggregation weight
        e_k = delta_k.bmm(curr_inten.reshape(-1, self.D, 1)) / self.tau
        e_k_u = F.softmax(e_k.squeeze(2), dim=1)
        v_u = e_k_u.unsqueeze(2).mul(delta_k).sum(dim=1)
        seq_output = F.normalize(v_u, p=2, dim=1)#参考CORE, 通过余弦度量距离

        return seq_output, curr_inten, new_s_u

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        seq_output, inten, new_s_u = self.forward(item_seq, 1)
        pos_items = interaction[self.POS_ITEM_ID]
        pos_items_emb = self.item_embedding(pos_items)

        real_s_u = torch.matmul(pos_items_emb, self.C_embedding.weight.transpose(0,1))
        idx = real_s_u.argsort(1)[:, -self.k:]
        real_C_u = self.C_embedding(idx)
        real_inten = self.gen2(real_C_u)
        real_re = self.dis(real_inten)
        gen_re = self.dis(inten)

        if (self.epochs // self.num) % 7 == 2:
            for name, para in self.named_parameters():
                if 'gen2' in name or 'embedding' in name:
                    para.requires_grad = True
                else:
                    para.requires_grad = False
            s_u = F.softmax(real_s_u/0.6, dim=1)
            selected = s_u.sort(1)[0][:, -self.k:]
            sup_selected = s_u.sort(1)[0][:, self.k:]
            G2_loss = -torch.mean(torch.log(selected+1e-8))
            return self.G2lr * G2_loss.requires_grad_(True)

        elif (self.epochs // self.num) % 7 == 1:
            for name, para in self.named_parameters():
                if 'dis' in name:
                    para.requires_grad = True
                else:
                    para.requires_grad = False
            D_loss = -torch.mean(torch.log(real_re+1e-8) + torch.log(1.-gen_re+1e-8))
            return self.Dlr * D_loss

        elif (self.epochs // self.num) % 7 == 3:
            for name, para in self.named_parameters():
                if 'gen1' in name or 'transfer' in name:
                    para.requires_grad = True
                else:
                    para.requires_grad = False
            G1_loss = -torch.mean(torch.nn.CosineSimilarity()(inten,real_inten))
            G1_loss = G1_loss + self.gama*torch.mean(torch.log(1.-gen_re+1e-8)) + -torch.mean((new_s_u-real_s_u)**2)
            return self.G1lr * G1_loss

        for name, para in self.named_parameters():
            if 'dis' not in name and 'C' not in name and 'gen2' not in name:
                para.requires_grad = True
            else:
                para.requires_grad = False
        all_item_emb = self.item_embedding.weight
        all_item_emb = F.normalize(all_item_emb, dim=-1)
        logits = torch.matmul(seq_output, all_item_emb.transpose(0, 1)) / self.temperature
        loss = self.loss_fct(logits, pos_items)
        
        return loss

    def predict(self, interaction):
        pass

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        seq_output, inten, new_s_u = self.forward(item_seq, 0)
        test_item_emb = self.item_embedding.weight
        # no dropout for evaluation
        test_item_emb = F.normalize(test_item_emb, dim=-1)
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        return scores



class GENERATOR1(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()

        # load parameters info
        self.D = config['embedding_size']
        self.device = config['device']
        self.loss_type = config['loss_type']
        self.layer_norm_eps = config['layer_norm_eps']
        self.initializer_range = config['initializer_range']
        self.LNorm = nn.LayerNorm(self.D, eps=self.layer_norm_eps)
        self.w1 = self._init_weight((self.D,self.D))
        self.w2 = self._init_weight(self.D)

    def forward(self, C_u, isTrain):
        if isTrain:
            noise2 = 0.05*torch.randn(C_u.shape).to(self.device)
        else:
            noise2 = 0 
        C_u_norm = self.LNorm(C_u + noise2)
        x = torch.matmul(C_u_norm, self.w1)
        x = torch.tanh(x)
        x = torch.matmul(x, self.w2)
        his_atten = F.softmax(x, dim=1)
        z_u = torch.matmul(his_atten.unsqueeze(2).transpose(1, 2), C_u_norm).transpose(1, 2)
        curr_inten = F.normalize(z_u.squeeze(2),p=2,dim=1)
        return curr_inten

    def _init_weight(self, shape):
        mat = np.random.normal(0, self.initializer_range, shape)
        return nn.Parameter(torch.FloatTensor(mat)).to(self.device)


class GENERATOR2(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()

        # load parameters info
        self.D = config['embedding_size']
        self.device = config['device']
        self.loss_type = config['loss_type']
        self.layer_norm_eps = config['layer_norm_eps']
        self.initializer_range = config['initializer_range']
        self.LNorm = nn.LayerNorm(self.D, eps=self.layer_norm_eps)
        self.w1 = self._init_weight((self.D,self.D))
        self.w2 = self._init_weight(self.D)

    def forward(self, real_C_u):
        real_C_u_norm = self.LNorm(real_C_u)
        x = torch.matmul(real_C_u_norm, self.w1)
        x = torch.tanh(x)
        x = torch.matmul(x, self.w2)
        new_atten = F.softmax(x, dim=1)
        z_u = torch.matmul(new_atten.unsqueeze(2).transpose(1, 2), real_C_u_norm).transpose(1, 2)
        real_inten = F.normalize(z_u.squeeze(2),p=2,dim=1)
        return real_inten

    def _init_weight(self, shape):
        mat = np.random.normal(0, self.initializer_range, shape)
        return nn.Parameter(torch.FloatTensor(mat)).to(self.device)


class DISCRIMINATOR(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()

        # load parameters info
        self.D = config['embedding_size']
        self.device = config['device']
        self.initializer_range = config['initializer_range']
        self.dis=nn.Sequential(
            nn.Linear(self.D, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, inten):
        result = self.dis(inten)
        return result
