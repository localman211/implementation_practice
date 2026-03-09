import torch
import torch.nn as nn

class CLFM_SGD(nn.Module):
    def __init__(self, user_cnt_list: list,
                    item_cnt_list: list,
                    user_k: int,
                    item_k: list,
                    common_k: int
                    ):

        super().__init__()
        self.user_cnt_list = user_cnt_list
        self.item_cnt_list = item_cnt_list
        self.domain_cnt = len(self.user_cnt_list)

        self.user_k = user_k
        self.item_k = item_k
        self.common_k = common_k
        # calculate domain-specific cluster number using common k and item k
        self.domain_k = [i_k - common_k for i_k in self.item_k]

        self.user_embs = nn.ModuleList()
        self.item_embs = nn.ModuleList()
        self.S0 = nn.Parameter(torch.empty(user_k, common_k))
        self.St = nn.ParameterList()

        torch.nn.init.normal_(self.S0, mean=0.0, std=0.001)

        # init embeddings (user, item, St) by domain
        for domain_idx in range(self.domain_cnt):
            self.user_embs.append(
                nn.Embedding(self.user_cnt_list[domain_idx], self.user_k)
            )
            self.item_embs.append(
                nn.Embedding(self.item_cnt_list[domain_idx], self.item_k[domain_idx])
            )
            self.St.append(
                nn.Parameter(torch.empty(self.user_k, self.domain_k[domain_idx]))
            )
            
            torch.nn.init.normal_(self.user_embs[domain_idx].weight, mean=0.0, std=0.001)
            torch.nn.init.normal_(self.item_embs[domain_idx].weight, mean=0.0, std=0.001)
            torch.nn.init.normal_(self.St[domain_idx], mean=0.0, std=0.01)

    def forward(self, data):
        pred = []

        # data shape (batch_size, domain_count, pair)
        for domain_idx in range(self.domain_cnt):
            user_ids, item_ids = data[domain_idx]

            user = self.user_embs[domain_idx](user_ids)
            item = self.item_embs[domain_idx](item_ids)
            S = torch.cat([self.S0, self.St[domain_idx]], dim=1)
            
            pred.append(torch.sum((user @ S) * item, dim=-1))
        
        pred = torch.stack(pred, dim=0)
        return pred

