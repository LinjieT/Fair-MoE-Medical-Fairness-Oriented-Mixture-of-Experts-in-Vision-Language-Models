import torch
import torch.nn as nn
import clip
import torch.nn.functional as F
from collections import OrderedDict
# from mixture_of_experts import MoE


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Expert(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Expert, self).__init__()
        self.network = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(input_size, input_size * 4)),
            ("gelu",  nn.ReLU()),  # ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(input_size * 4, input_size))
        ]))

    def forward(self, x):
        return self.network(x)


class MoE(nn.Module):
    def __init__(self, num_experts, input_size, output_size):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.gate = nn.Sequential(
            nn.Linear(input_size, int(input_size/4)),
            QuickGELU(),
            nn.Linear(int(input_size/4), num_experts)
        )
        # nn.Linear(input_size, num_experts)
        self.experts = nn.ModuleList(
            [Expert(input_size, output_size) for _ in range(num_experts)])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 门控网络的输出，确定各专家的权重
        # gating_weights = F.softmax(self.gate(x), dim=1)
        Output_weight = self.gate(x)
        gating_weights = self.softmax(Output_weight)
        # 专家网络的输出
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # 将输出堆叠成一个新的维度

        # 加权求和专家的输出
        output = torch.bmm(gating_weights.unsqueeze(1),
                           expert_outputs).squeeze(1)

        # Residual
        x = x+output

        return output, Output_weight


class FairTokenMoE(nn.Module):
    def __init__(self, num_experts, input_size, output_size, k, capacity_rate):
        super(FairTokenMoE, self).__init__()
        self.num_experts = num_experts
        self.k = k  # Top k Expert
        self.apacity_rate = capacity_rate
        self.capacity = int(197*capacity_rate*k/num_experts)

        self.gate = nn.Sequential(
            nn.Linear(input_size, int(input_size/4)),
            nn.ReLU(),  # QuickGELU(),
            nn.Linear(int(input_size/4), num_experts)
        )
        self.experts = nn.ModuleList(
            [Expert(input_size, output_size) for _ in range(num_experts)])
        self.softmax = nn.Softmax(dim=2)
        self.OutWeight = nn.Identity()

    def forward(self, x):
        # input: Tokens(197) x Batchsize x Dim(768)
        # Output_weight: Tokens x Batchsize x NExp(num_experts)
        Output_weight = self.gate(x)
        gating_weights = self.softmax(Output_weight)

        # expert(x): Tokens(197) x Batchsize x Dim(768)
        expert_outputs = [expert(x) for expert in self.experts]

        # Combine Expert output together
        # expert_outputs:  Tokens(197) x Batchsize x NExp(num_experts) x Dim(768)
        expert_outputs = torch.stack(expert_outputs, dim=2)

        # Choose top k expert
        topk_values, topk_indices = torch.topk(gating_weights, self.k, dim=2)

        # Mask to filter out not chosen expert
        MaskExp = torch.zeros_like(
            gating_weights, dtype=torch.bool).scatter_(2, topk_indices, True)
        ChosenExpertWeight = gating_weights * MaskExp

        # Choose in capacity token for each expert
        topk_values1, topk_indices1 = torch.topk(
            ChosenExpertWeight, self.capacity, dim=0)

        # Mask out of capacity tokens
        MaskCapacity = torch.zeros_like(
            ChosenExpertWeight, dtype=torch.bool).scatter_(0, topk_indices1, True)
        # FinalWeight: Tokens x Batchsize x NExp(num_experts)
        FinalWeight = ChosenExpertWeight * MaskCapacity

        # To output weight using hook
        FinalWeight = self.OutWeight(FinalWeight)

        # Combine weight and outputs of experts
        # expert_outputs:  Tokens(197) x Batchsize x NExp(num_experts) x Dim(768)
        # FinalWeight: Tokens x Batchsize x NExp(num_experts)
        # output[t,b,i,:] = expert_outputs[t,b,i,:]*FinalWeight[t,b,i], scalar product
        output = torch.einsum('tbnf, tbn -> tbf', expert_outputs, FinalWeight)
        # No Residual
        output = output - x

        #

        # 加权求和专家的输出
        # output = torch.bmm(gating_weights.unsqueeze(1), expert_outputs).squeeze(1)

        # Residual
        # x=x+output

        return output


class FairMoE(nn.Module):
    def __init__(self, OriginalModel, size, Nexpert):
        super(FairMoE, self).__init__()
        # OriginalModel, preprocess = clip.load('ViT-B/16', device='cuda:0', jit=False)
        # For IMG encoder
        self.visual = OriginalModel.visual
        # For Text encoder
        self.positional_embedding = OriginalModel.positional_embedding
        self.token_embedding = OriginalModel.token_embedding
        self.transformer = OriginalModel.transformer
        self.ln_final = OriginalModel.ln_final
        self.text_projection = OriginalModel.text_projection
        # For forwards
        self.logit_scale = OriginalModel.logit_scale

        par0 = OriginalModel.visual.transformer.resblocks[0].mlp.state_dict()
        par11 = OriginalModel.visual.transformer.resblocks[11].mlp.state_dict()
        par11TxT = OriginalModel.transformer.resblocks[11].mlp.state_dict()

        # For MoE
        if size == 'vit-b16':

            # abalation for img
            self.moe = MoE(Nexpert, 512, 2048).half()  # 768

            # self.tokenmoe0 = FairTokenMoE(10, 768, 768*4, 3, 0.8).half() # 768
            # for i in range(10):
            #   self.tokenmoe0.experts[i].network.load_state_dict(par0)

            # abalation for img
            self.tokenmoe11 = FairTokenMoE(
                10, 768, 768*4, 3, 0.8).half()  # 768
            for i in range(10):
                self.tokenmoe11.experts[i].network.load_state_dict(par11)

            # abalation for text
            self.tokenmoe11TxT = FairTokenMoE(
                10, 512, 512*4, 3, 0.8).half()  # 768
            for i in range(10):
                self.tokenmoe11TxT.experts[i].network.load_state_dict(par11TxT)

            # abalation for text
            self.moetext = MoE(Nexpert, 512, 2048).half()  # 768
            # self.TxTtoken

        else:

            # 768 #abalation for img
            self.moe = MoE(Nexpert, 768, 768*4).half()

            # self.tokenmoe0 = FairTokenMoE(5, 1024, 1024*4, 3, 0.8).half() # 768
            # for i in range(5):
            #   self.tokenmoe0.experts[i].network.load_state_dict(par0)

            # abalation for img
            self.tokenmoe11 = FairTokenMoE(
                10, 1024, 1024*4, 3, 0.8).half()  # 768
            for i in range(10):
                self.tokenmoe11.experts[i].network.load_state_dict(par11)

            # abalation for text
            self.tokenmoe11TxT = FairTokenMoE(
                10, 768, 768*4, 3, 0.8).half()  # 768
            for i in range(10):
                self.tokenmoe11TxT.experts[i].network.load_state_dict(par11TxT)

            # abalation for text
            self.moetext = MoE(Nexpert, 768, 768*4).half()  # 768

        # Add token MoE to Visual Encoder
        # self.visual.transformer.resblocks[0].mlp = self.tokenmoe0

        # Ablation to remove
        self.visual.transformer.resblocks[-1].mlp = self.tokenmoe11
        # Ablation to remove
        self.transformer.resblocks[-1].mlp = self.tokenmoe11TxT

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        image_features = self.visual(image.type(self.dtype))
        image_features, image_moe_weight = self.moe(
            image_features)  # for abalation for removing img
        # image_moe_weight = 0
        return image_features,  image_moe_weight
        # return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(
            self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)
              ] @ self.text_projection

        # feature moe
        x, text_moe_weight = self.moetext(x)  # for abalation for removing text
        # text_moe_weight = 0

        return x, text_moe_weight

    def forward(self, image, text):
        image_features, image_moe_weight = self.encode_image(image)
        text_features, text_moe_weight = self.encode_text(text)
        # print(' no MoE Image Norm Value:',image_features.norm(dim=1, keepdim=True))

        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # MoE layers
        # image_features = image_features.unsqueeze(0)
        # image_features = image_features.float()
        # text_features = text_features.unsqueeze(0)
        # text_features = text_features.float()

        # image_features, aux_loss = self.moe(image_features)
        # print("image_features shape (after moe): ", image_features.shape)
        # print("text feature (no moe): ", text_features.shape)
        #   text_features, _ = self.moe(text_features)
        # image_features = image_features[0]
        # text_features = text_features[0]

        # normalized features
        image_features = image_features / \
            image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # print('Image Norm Value:',image_features.norm(dim=1, keepdim=True))
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        # , aux_loss #No auxiliary loss for MoE2.0
        return logits_per_image, logits_per_text, image_moe_weight, text_moe_weight
