import torch
from typing import Union
from torch.nn import functional as F

class ContrastLoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature


class RetriContrastLoss(ContrastLoss):
    '''
    loss for retrieval
    if use_all_pair is set to true, it will use the query-query pair as neg, 
    otherwise it use query-passage as neg
    '''
    def __init__(self, temperature: float = 0.05, use_all_pair=False, mixcse=-1):
        super().__init__()
        self.temperature = temperature
        self._cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.use_all_pair=use_all_pair
        self.mixcse = mixcse

    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_pos_embeddings: torch.Tensor,
        text_neg_embeddings: Union[torch.Tensor, None],
        text_neg_index: Union[torch.Tensor, None],
    ) -> torch.Tensor:
        if self.mixcse > 0:
            assert text_neg_embeddings is not None
            assert text_neg_embeddings.size(0) == text_embeddings.size(0)
        
        if text_neg_embeddings is None:
            sim_matrix = torch.cosine_similarity(
                text_embeddings.unsqueeze(1),
                text_pos_embeddings.unsqueeze(0),
                dim=-1,
            )
            sim_matrix = sim_matrix / self.temperature
            labels = torch.arange(sim_matrix.size(0), device=text_embeddings.device, dtype=torch.long)
            loss = self._cross_entropy_loss(sim_matrix, labels)
            return loss
        
        sim_neg_matrix = torch.cosine_similarity(
            text_embeddings.unsqueeze(1),
            text_neg_embeddings.unsqueeze(0),
            dim=-1,
        )
        
        if self.use_all_pair:
            sim_pos_matrix = torch.cosine_similarity(
                text_embeddings.unsqueeze(1),
                text_pos_embeddings.unsqueeze(0),
                dim=-1,
            )
            sim_matrix = torch.cat([sim_pos_matrix, sim_neg_matrix], dim=1)
            labels = torch.arange(sim_matrix.size(0), dtype=torch.long, device=sim_matrix.device)
        else:
            sim_pos_vector = torch.cosine_similarity(text_embeddings, text_pos_embeddings, dim=-1)
            sim_matrix = torch.cat([sim_pos_vector.unsqueeze(1), sim_neg_matrix], dim=1)
            labels = torch.zeros(sim_matrix.size(0), dtype=torch.long, device=sim_matrix.device)
        
        if self.mixcse > 0:
            # constuct hard negative sample
            lambdas = [self.mixcse] * text_embeddings.size(0)
            lambdas = torch.tensor(lambdas).to(text_embeddings.device)
            hard_embeddings = lambdas.unsqueeze(1) * text_pos_embeddings + (1 - lambdas).unsqueeze(1) * text_neg_embeddings
            hard_embeddings = hard_embeddings.detach()
            hard_cos = torch.cosine_similarity(text_embeddings, hard_embeddings, dim=-1)
            sim_matrix = torch.cat([sim_matrix, hard_cos.unsqueeze(1)], dim=1)
            # lambdas = [self.mixcse] * text_embeddings.size(0)
            # lambdas = torch.tensor(lambdas).to(text_embeddings.device)
            # neg_embeddings = torch.cat([text_pos_embeddings[1:],text_pos_embeddings[:1]],dim=0)
            # # neg_embeddings = text_neg_embeddings
            # hard_embeddings = lambdas.unsqueeze(1) * text_pos_embeddings + (1 - lambdas).unsqueeze(1) * neg_embeddings
            # hard_embeddings = hard_embeddings.detach()
            # hard_cos = torch.cosine_similarity(text_embeddings.unsqueeze(1), hard_embeddings.unsqueeze(0), dim=-1)
            # mask = torch.eye(hard_cos.size(0)).to(hard_cos.device)
            # mask = torch.cat([mask[-1:],mask[:-1]])
            # mask = 1 - mask
            # hard_cos = mask * hard_cos
            # sim_matrix = torch.cat([sim_matrix, hard_cos], dim=1)
            
            # neg_embeddings2 = torch.cat([text_embeddings[1:], text_embeddings[:1]], dim=0)
            # # neg_embeddings2 = text_neg_embeddings
            # hard_embeddings2 = lambdas.unsqueeze(1) * text_embeddings + (1 - lambdas).unsqueeze(1) * neg_embeddings2
            # hard_embeddings2 = hard_embeddings2.detach()
            # hard_cos2 = torch.cosine_similarity(hard_embeddings2.unsqueeze(1), text_pos_embeddings.unsqueeze(0), dim=-1)
            # mask2 = torch.eye(hard_cos2.size(0)).to(hard_cos2.device)
            # mask2 = torch.cat([mask2[:,-1:], mask2[:,:-1]], dim=1)
            # mask2 = 1 - mask2
            # hard_cos2 = mask2 * hard_cos2
            # sim_matrix = torch.cat([sim_matrix, hard_cos2], dim=1)
            
        # sim_matrix = sim_matrix / self.temperature
        # loss = self._cross_entropy_loss(sim_matrix, labels)
        ## for circle_loss
        m = 0.5
        gamma = 256
        labels = F.one_hot(labels)
        if labels.size(1) != sim_matrix.size(1):
            neg_lbl = torch.zeros(labels.size(0), sim_matrix.size(1)-labels.size(1), dtype=torch.long, device=sim_matrix.device)
            labels = torch.cat([labels, neg_lbl], dim=1)
        assert labels.size(1) == sim_matrix.size(1) and labels.size(0) == sim_matrix.size(0), f"{labels.size(1)} {sim_matrix.size(1)} {labels.size(0)} {sim_matrix.size(0)}"
        sim_neg_matrix = sim_matrix[labels == 0]
        an = torch.clamp_min(sim_neg_matrix.detach() + (1+m), min=0.0)
        delta_n = -(1-m)
        sim_neg_matrix = an * (sim_neg_matrix - delta_n)

        sim_pos_matrix = sim_matrix[labels == 1]
        ap = torch.clamp_min(- sim_pos_matrix.detach() + (1+m), min=0.0)
        delta_p = 1 - m
        sim_pos_matrix = - ap * (sim_pos_matrix - delta_p)
        loss = torch.nn.Softplus()(torch.logsumexp(sim_neg_matrix, dim=0) + torch.logsumexp(sim_pos_matrix, dim=0))
        return loss


class RetriCoSentLoss(ContrastLoss):
    bias: torch.Tensor

    def __init__(self, temperature: float = 0.05, cosent_w: float = 0.04) -> None:
        super().__init__(temperature)
        self.register_buffer('bias', torch.tensor([0.0]))
        self.cosent_w = cosent_w
        
    def forward(self, text_embeddings: torch.Tensor, text_pair_embeddings: torch.Tensor, true_similarity: torch.Tensor) -> torch.Tensor:
        bs = text_embeddings.shape[0]
        assert text_pair_embeddings.shape[0] % bs == 0, f"neg num is not equal for each sample: {bs}, {text_pair_embeddings.shape[0]}"
        pair_num = int(text_pair_embeddings.shape[0] // bs)
        ### v3
        # text_pair_embeddings = text_pair_embeddings.view(bs, pair_num, -1)
        # norm_embed1 = F.normalize(text_embeddings, p=2, dim=1, eps=1e-8)
        # norm_embed2 = F.normalize(text_pair_embeddings, p=2, dim=2, eps=1e-8)
        # sim = torch.sum(norm_embed1.unsqueeze(1) * norm_embed2, dim=2) * 20
        # sim = sim.view(-1)
        # sim = sim[:, None] - sim[None, :]
        # label = true_similarity[:, None] < true_similarity[None, :]
        # label = label.float()
        # sim = sim - (1 - label) * 1e12
        # sim = torch.cat((torch.zeros(1).to(sim.device), sim.view(-1)), dim=0)
        # loss = torch.logsumexp(sim, dim=0)
        ### v2
        text_pair_embeddings = text_pair_embeddings.view(bs, pair_num, -1)
        true_similarity = true_similarity.view(bs, pair_num)
        
        norm_embed1 = F.normalize(text_embeddings, p=2, dim=1, eps=1e-8)
        norm_embed2 = F.normalize(text_pair_embeddings, p=2, dim=2, eps=1e-8)
        sim = torch.sum(norm_embed1.unsqueeze(1) * norm_embed2, dim=2) * 20
        sim = sim[:, :, None] - sim[:, None, :]
        # #### lambda rank
        # label_diffs = true_similarity[:, :, None] - true_similarity[:, None, :]
        # sim = (sim - sim.min()) / (sim.max() - sim.min()) * 4 - 2    ### 这里的值域范围受限，是否会导致即便已经训练到最优值，仍然还有梯度
        # lambda_ij = torch.abs(
        #     (2 ** true_similarity[:, :, None] - 1) / torch.log2(torch.arange(2, true_similarity[0].numel()+2, device=sim.device)[None, None, :]) -
        #     (2 ** true_similarity[:, None, :] - 1) / torch.log2(torch.arange(2, true_similarity[0].numel()+2, device=sim.device)[None, :, None])
        # )
        # loss = lambda_ij * (1- torch.sigmoid(label_diffs * sim))
        # loss = loss.mean()
        label = true_similarity[:, :, None] < true_similarity[:, None, :]
        label = label.float()
        sim = sim - (1 - label) * 1e12
        sim = torch.cat((torch.zeros(1).to(sim.device), sim.view(-1)), dim=0)
        loss = torch.logsumexp(sim, dim=0)
        ### v1
        # predict_similarity = torch.cosine_similarity(text_embeddings.unsqueeze(1), text_pair_embeddings, dim=-1) / self.temperature
        # cosine_similarity_diff = predict_similarity.unsqueeze(2) - predict_similarity.unsqueeze(1)
        # smaller_mask = true_similarity.unsqueeze(2) < true_similarity.unsqueeze(1)
        # cosine_similarity_diff = cosine_similarity_diff[smaller_mask]
        # cosine_diff_scores_add_bias = torch.cat((cosine_similarity_diff, self.bias))
        # loss = torch.logsumexp(cosine_diff_scores_add_bias, dim=0) * self.cosent_w
        return loss


class CoSentLoss(ContrastLoss):
    '''
    loss for sts and pair classification.
    here we hard code the cosent loss weight to 0.04
    '''
    bias: torch.Tensor

    def __init__(self, temperature: float = 0.05, cosent_w: float = 0.04) -> None:
        super().__init__(temperature)
        self.register_buffer('bias', torch.tensor([0.0]))
        self.cosent_w = cosent_w

    def forward(self, predict_similarity: torch.Tensor, true_similarity: torch.Tensor) -> torch.Tensor:
        predict_similarity = predict_similarity / self.temperature
        cosine_similarity_diff = -(predict_similarity.unsqueeze(0) - predict_similarity.unsqueeze(1))
        smaller_mask = true_similarity.unsqueeze(0) <= true_similarity.unsqueeze(1)
        cosine_similarity_diff = cosine_similarity_diff[~smaller_mask]
        cosine_diff_scores_add_bias = torch.cat((cosine_similarity_diff, self.bias))
        loss = torch.logsumexp(cosine_diff_scores_add_bias, dim=0) * self.cosent_w
        return loss

class ClsContrastLoss(torch.nn.Module):
    '''
    loss for clustering and classification
    here we hard code the cls contrast loss weight to 0.2
    '''
    def __init__(self, temperature: float = 0.05, cls_w = 0.2):
        super().__init__()
        self.temperature = temperature
        self._cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.cls_w = cls_w
    
    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_pos_embeddings: torch.Tensor,
        text_neg_embeddings: torch.Tensor,
        ) -> torch.Tensor:
        bs = text_embeddings.shape[0]
        assert text_neg_embeddings.shape[0] % bs == 0, f"neg num is not equal for each sample: {bs}, {text_neg_embeddings.shape[0]}"
        neg_num = int(text_neg_embeddings.shape[0] // bs)

        sim_neg_matrix = torch.cosine_similarity(
            text_embeddings.unsqueeze(1),
            text_neg_embeddings.unsqueeze(0),
            dim=-1,
        )
        sim_pos_vector = torch.cosine_similarity(text_embeddings, text_pos_embeddings, dim=-1)

        # find the neg for eatch training sample
        neg_matrix = []
        for i in range(bs):
            neg_matrix.append(sim_neg_matrix[i, i * neg_num : (i + 1) * neg_num])
        sim_neg_matrix = torch.stack(neg_matrix)
        sim_matrix = torch.cat([sim_pos_vector.unsqueeze(1), sim_neg_matrix], dim=1)
        sim_matrix = sim_matrix / self.temperature
        labels = torch.zeros(sim_matrix.size(0), dtype=torch.long, device=sim_matrix.device)
        loss = self._cross_entropy_loss(sim_matrix, labels) * self.cls_w
        return loss
