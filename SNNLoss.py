import torch
import torch.nn as nn


def euc_dist(x):
    return torch.norm(x[:, None] - x, dim=2, p=2)


def rbf_dist(x):
    return (x[:, None] - x).pow(2).sum(dim=2) / (2. * x.var())


def cosine_distance(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


# SNNLoss definition
class SNNLoss(nn.Module):

    def __init__(self, std=True, inv=False, eps=1e-6):
        super(SNNLoss, self).__init__()
        self.eps = eps
        self.std = std

    def forward(self, x, y, temp=0., d=None):  # x 2-D matrix of BxF, y 1-D vector of B
        """
        :param temp: float, log10 of the temperature
            e.g. if temp=0 ==> temperature=10^0
        """
        # T = 1/torch.tensor([100.]).to(x.device)

        # SE LA TEMPERATURA NON E' SPECIFICATA SETTALA A 0 -> 10^0=1
        temp = torch.tensor([float(temp)]).to(x.device)

        b = len(y)

        # print(f"Standard deviation: {x.std().item()}")

        #if self.std:
        #   x = x / x.std()  # x.std() = 0.122

        # print(f"Norm of features0 after standardization: {torch.norm(x[0, :])}")

        # MATRICE DI DISTANZA IN CUI:
        # - DIAGONALE: DISTANZA TRA L'ELEMENTO i-ESIMO E SE STESSO = 0
        # - FUORI DALLA DIAGONALE: DISTANZA TRA L'ELEMENTO i-ESIMO E j-ESIMO
        dist = rbf_dist(x)

        # make diagonal mask
        # MATRICE CON 0 SULLA DIAGONALE E 1 FUORI
        m_den = 1 - torch.eye(b)
        m_den = m_den.float().to(x.device)

        # MATRICE DI e^dist
        e_dist = (-dist) * torch.pow(10, temp)

        # MATRICE DI DISTANZE DEI DENOMINATORI
        den_dist = torch.clone(e_dist)

        # OGNI QUALVOLTA LA DISTANZA AL DENOMINATORE E' ZERO INSERISCO -inf
        # così facendo sto settando a -inf la distanza dei punti che non appartengono alla classe riferita alla riga
        den_dist[m_den == 0] = float('-inf')

        # make per class mask
        # PER OGNI RIGA (OGNI CLASSE) HO 1 SOLO SULLE IMMAGINI CHE APPARTENGONO A QUELLE CLASSI

        m_num = (y == y.unsqueeze(0).t()).type(torch.int) - torch.eye(b, dtype=torch.int).to(y.device)

        # print(m_num)
        num_dist = torch.clone(e_dist)

        # OGNI QUALVOLTA LA DISTANZA AL DENOMINATORE E' ZERO INSERISCO -inf
        # così facendo sto settando a -inf la distanza dei punti che non appartengono alla classe riferita alla riga
        num_dist[m_num == 0] = float('-inf')
        # print(num_dist)
        # compute logsumexp
        num = torch.logsumexp(num_dist, dim=1)
        den = torch.logsumexp(den_dist, dim=1)

        if torch.sum(torch.isinf(num)) > 0:
            num = num.clone()
            den = den.clone()
            den[torch.isinf(num)] = 0
            num[torch.isinf(num)] = 0
            # print(torch.bincount(y))

        if torch.sum(torch.isnan(num)) > 0:
            print(x.shape)
            print(x)
            print(num_dist.shape)
            print(num_dist)
            print(den_dist)
            print(num.shape)
            print(num)
            print(den)
            raise Exception()

        return -(num - den).mean()