import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    def __init__(
        self,
        in_feature: int,
        out_feature: int,
        s: float,
        m: float,
        easy_margin: bool = False,
    ):
        """[summary]

        Args:
            in_feature (int): input feature size
            out_feature (int): num_classes
            s (float): norm of input feature
            m (float): margin
            easy_margin (bool, optional): [description]. Defaults to False.
        """
        super(ArcMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.weight = Parameter(
            torch.Tensor(out_feature, in_feature),
        )
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(
                (cosine - self.th) > 0,
                phi,
                cosine - self.mm,
            )

        # convert label to one-hot
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        return output
