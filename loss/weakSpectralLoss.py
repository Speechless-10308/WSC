import torch.nn as nn
import torch


class WeakSpectralLoss(nn.Module):
    def __init__(self, alpha, beta, args=None):
        super(WeakSpectralLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        if args is not None:
            self.lam_consist = args.lam_consist

    def forward(self, zq1, zq2, q, z1=None, z2=None, pi=None):

        if pi is None:
            pi = (torch.ones(q.shape[1], 1) / q.shape[1]).cuda()

        Q = torch.matmul(q, q.t())
        W = torch.matmul(q, pi)
        W_double = torch.cat([W, W], dim=0)
        WxW = torch.matmul(W, W.t())
        if z1 is not None and z1.shape[0] >= 2:
            anti_identity_matrix_zq = (1 - torch.eye(zq1.shape[0])).cuda()
            anti_identity_matrix_z = (1 - torch.eye(z1.shape[0])).cuda()

            zq1xzq2 = torch.matmul(zq1, zq2.t())
            zq1xzq1 = torch.matmul(zq1, zq1.t())
            zq2xzq2 = torch.matmul(zq2, zq2.t())
            z1xz1 = torch.matmul(z1, z1.t())
            z2xz2 = torch.matmul(z2, z2.t())
            z1xz2 = torch.matmul(z1, z2.t())
            zqxz = torch.matmul(torch.cat([zq1, zq2], dim=0), torch.cat([z1, z2], dim=0).t())
            pow_zqxz = zqxz ** 2
            pow_zq1xzq2 = zq1xzq2 ** 2
            pow_z1xz2 = z1xz2 ** 2
            pow_z1xz1 = z1xz1 ** 2
            pow_z2xz2 = z2xz2 ** 2
            pow_zq1xzq1 = zq1xzq1 ** 2
            pow_zq2xzq2 = zq2xzq2 ** 2


            # compute l1
            l1 = -2 * self.alpha * (torch.trace(zq1xzq2) + torch.trace(z1xz2)) / (zq1.shape[0] + z1.shape[0])

            # compute l2
            l2_1 = -2 * self.beta * (torch.sum(zq1xzq2 * Q * anti_identity_matrix_zq)) / (
                    zq1.shape[0] * (zq1.shape[0] - 1))
            l2_2 = -2 * self.beta * (torch.sum(zq1xzq1 * Q * anti_identity_matrix_zq)) / (
                    zq1.shape[0] * (zq1.shape[0] - 1))
            l2_3 = -2 * self.beta * (torch.sum(zq2xzq2 * Q * anti_identity_matrix_zq)) / (
                    zq1.shape[0] * (zq1.shape[0] - 1))
            l2 = (l2_1 + l2_2 + l2_3) / 3

            # compute l3
            l3_1 = (self.alpha * self.alpha * (torch.sum(pow_zq1xzq2 * anti_identity_matrix_zq) +
                                               torch.sum(pow_z1xz2 * anti_identity_matrix_z))
                    / (zq1.shap[0] * (zq1.shape[0] - 1) + z1.shap[0] * (z1.shape[0] - 1)))
            l3_2 = (self.alpha * self.alpha * (torch.sum(pow_zq1xzq1 * anti_identity_matrix_zq) +
                                               torch.sum(pow_z1xz1 * anti_identity_matrix_z))
                    / (zq1.shap[0] * (zq1.shape[0] - 1) + z1.shap[0] * (z1.shape[0] - 1)))
            l3_3 = (self.alpha * self.alpha * (torch.sum(pow_zq2xzq2 * anti_identity_matrix_zq) +
                                               torch.sum(pow_z2xz2 * anti_identity_matrix_z))
                    / (zq1.shap[0] * (zq1.shape[0] - 1) + z1.shap[0] * (z1.shape[0] - 1)))
            l3 = (l3_1 + l3_2 + l3_3) / 3

            # compute l4

            l4_1 = (self.beta * self.beta * torch.sum(pow_zq1xzq2 * anti_identity_matrix_zq * WxW)
                    / (zq1.shape[0] * (zq1.shape[0] - 1)))
            l4_2 = (self.beta * self.beta * torch.sum(pow_zq1xzq1 * anti_identity_matrix_zq * WxW)
                    / (zq1.shape[0] * (zq1.shape[0] - 1)))
            l4_3 = (self.beta * self.beta * torch.sum(pow_zq2xzq2 * anti_identity_matrix_zq * WxW)
                    / (zq1.shape[0] * (zq1.shape[0] - 1)))
            l4 = (l4_1 + l4_2 + l4_3) / 3

            # compute l5
            l5 = (self.beta * self.alpha * 2 *
                  torch.sum(
                      torch.diag_embed(W_double[:, 0]) @ pow_zqxz + (torch.diag_embed(W[:, 0]) @ pow_zq1xzq2) * anti_identity_matrix_zq)
                  / (4 * zq1.shape[0] * z1.shape[0] + zq1.shape[0] * (zq1.shape[0] - 1)))

            loss = l1 + l2 + self.lam_consist * (l3 + l4 + l5)


        else:
            anti_identity_matrix_zq = (1 - torch.eye(zq1.shape[0])).cuda()
            anti_identity_matrix_za = (1 - torch.eye(2 * zq1.shape[0])).cuda()
            zq1xzq2 = torch.matmul(zq1, zq2.t())
            zq1xzq1 = torch.matmul(zq1, zq1.t())
            zq2xzq2 = torch.matmul(zq2, zq2.t())
            zaxza = torch.matmul(torch.cat([zq1, zq2], dim=0), torch.cat([zq1, zq2], dim=0).t())
            pow_zaxza = zaxza ** 2
            # pow_zq1xzq1 = zq1xzq1 ** 2
            # pow_zq2xzq2 = zq2xzq2 ** 2
            # pow_zq1xzq2 = zq1xzq2 ** 2
            #
            l1 = -2 * self.alpha * torch.trace(zq1xzq2) / zq1.shape[0]

            l2_1 = -2 * self.beta * (torch.sum(zq1xzq2 * Q * anti_identity_matrix_zq)) / (
                    zq1.shape[0] * (zq1.shape[0] - 1))
            l2_2 = -2 * self.beta * (torch.sum(zq1xzq1 * Q * anti_identity_matrix_zq)) / (
                    zq1.shape[0] * (zq1.shape[0] - 1))
            l2_3 = -2 * self.beta * (torch.sum(zq2xzq2 * Q * anti_identity_matrix_zq)) / (
                    zq1.shape[0] * (zq1.shape[0] - 1))
            l2 = (l2_1 + l2_2 + l2_3) / 3

            l3_new = self.lam_consist *  (self.alpha + self.beta / q.shape[1]) * (self.alpha + self.beta / q.shape[1])  * torch.sum(pow_zaxza * anti_identity_matrix_za) / (2 * zq1.shape[0] * (2 * zq1.shape[0] - 1))
            # loss = l1 + l2 + l3 + l4 + l5
            loss = l1 + l2 + l3_new

        return loss, l1, l2
