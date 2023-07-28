from turtle import forward
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from metrics.mmd import MMD_loss


class MarginStep(object):
    def __init__(self, margin_min=0.5, margin_max=5.0, step_num=4, epochs=10000):
        """initialize a margin step scheduler.
        Args:
            margin_min (float): min margin.
            margin_max (flaot): max margin.
            step_num (int): num of scheduler stps.
            epochs (int): num of all epochs.
        """
        self.margin_min = margin_min
        self.margin_max = margin_max
        self.step_num = step_num
        self.epochs = epochs

    def get_margin(self, epoch):
        """step the margin according to the epoch.
        Args:
            epoch (train epoch): 
        Returns:
            _type_: Current Margin.
        """
        assert 0 <= epoch <= self.epochs, "epoch invalid!"
        part_epoch = self.epochs // self.step_num
        part_margin = (self.margin_max - self.margin_min) / (self.step_num - 1)
        for i in range(1, self.step_num + 1):
            if epoch <= i * part_epoch:
                return (i - 1) * part_margin + self.margin_min


class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # if no margin assigned, use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda: y = y.cuda()
            ap_dist = torch.norm(anchor-pos, 2, dim=1).view(-1)
            an_dist = torch.norm(anchor-neg, 2, dim=1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)
        return loss


class MMD_TripletLoss(nn.Module):
    """ compute MMD Thriplet Loss according anchor, positivte, negative.
    """
    def __init__(self, margin, kermel_type='rbf'):
        super(MMD_TripletLoss, self).__init__()
        self.margin = margin
        self.Loss = MMD_loss(kermel_type)

    def forward(self, anc, pos, neg):
        anc_pos_dist = self.Loss(anc, pos)
        anc_neg_dist = self.Loss(anc, neg)
        zero_ = torch.tensor(0)
        if anc.is_cuda: zero_ = zero_.cuda()
        loss = max((anc_pos_dist - anc_neg_dist + self.margin), zero_)
        return loss


class COSSIM_TripletLoss(nn.Module):
    """ compute Cosine Similarity Thriplet Loss according anchor, positivte, negative.
    """
    def __init__(self, margin):
        super(COSSIM_TripletLoss, self).__init__()
        self.margin = margin

    def cosine_similarity(self, x1, x2):
        """cosine similarity is [-1, 1].
            smaller is nonsimilarity, 1 is most similarity.
        """
        return -torch.cosine_similarity(x1, x2)

    def forward(self, anchor, pos, neg):
        Loss = nn.TripletMarginWithDistanceLoss(distance_function=self.cosine_similarity, margin=self.margin)
        return Loss(anchor, pos, neg)


class HYPERSPHERE_TripletLoss(nn.Module):
    """ compute HyperSphere Distance Thriplet Loss according anchor, positivte, negative.
    """
    def __init__(self, margin):
        super(HYPERSPHERE_TripletLoss, self).__init__()
        self.margin = margin

    def project_to_hypersphere(self, p):
        """ Project `p` to the sphere points `x & y`.
            Following formula: (p, 0) -> x=2p/(p^2+1), y=(p^2-1)/(p^2+1)
            real/fake output from (B, c) ro (B, c+1).
            the same as TensorFlow code.
        """
        p_norm = torch.norm(p, dim=1, keepdim=True) ** 2
        x = 2 * p / (p_norm + 1)
        y = (p_norm - 1) / (p_norm + 1)
        return torch.cat([x, y], dim=1)

    def hypersphere_distance(self, x1, x2, moment=5, eps=1e-6):
        '''
        Calcuate distance between x1 and x2 using hypersphere metrics. Matching monents from 1-3.
        ds(p, q) = acos((p2p2 -p2 -q2 + 4pq + 1) / (p2+1)(q2+1)).
        '''
        p = self.project_to_hypersphere(x1)
        q = self.project_to_hypersphere(x2)
        p_norm = torch.norm(p, dim=1) ** 2
        q_norm = torch.norm(q, dim=1) ** 2
        top = p_norm * q_norm - p_norm - q_norm + torch.sum(4 * p * q, dim=1) + 1  # (b, 1)
        bottom = (p_norm + 1) * (q_norm + 1)  # (b, 1)
        sphere_d = torch.acos((top / bottom).clamp_(-1.0+eps, 1.0-eps)) # ds(p, q), clamp to (-1, 1).
        distance = 0
        for i in range(1, moment + 1):
            distance += torch.pow(sphere_d, i)  # (b, 1), do not calculate mean
        return distance

    # Simplify as follows:
    # def hypersphere_distance(self, x1, x2, moment=5, eps=1e-6):
    #     '''
    #     Calcuate distance between x1 and x2 using hypersphere metrics. Matching monents from 1-3.
    #     ds(p, q) = acos((p2p2 -p2 -q2 + 4pq + 1) / (p2+1)(q2+1)).
    #     '''
    #     p = self.project_to_hypersphere(x1)
    #     q = self.project_to_hypersphere(x2)
    #     sphere_d = torch.acos(torch.sum(p * q, dim=1).clamp_(-1.0+eps, 1.0-eps)) # ds(p, q), clamp to (-1, 1).
    #     distance = 0
    #     for i in range(1, moment + 1):
    #         distance += torch.pow(sphere_d, i)  # (b, 1), do not calculate mean
    #     return distance

    def forward(self, anchor, pos, neg):
        Loss = nn.TripletMarginWithDistanceLoss(distance_function=self.hypersphere_distance, margin=self.margin)
        return Loss(anchor, pos, neg)


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    anc = torch.rand(10, 256).cuda()
    pos = torch.rand(10, 256).cuda()
    neg = torch.rand(10, 256).cuda()

    loss2 = COSSIM_TripletLoss(margin=0.3)
    print(loss2(anc, pos, neg))
