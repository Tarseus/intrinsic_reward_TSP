
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem, EMBEDDING_DIM)
        self.qkv_dim = model_params['qkv_dim']
        self.head_num = model_params['head_num']

        self.problem_size = model_params['problem_size']

    def pre_forward(self, reset_state, self_rs_decoder = None):
        self.encoded_nodes = self.encoder(reset_state.problems)
        # shape: (batch, problem, EMBEDDING_DIM)
        self.decoder.set_kv(self.encoded_nodes)
        if self_rs_decoder is not None:
            self_rs_decoder.set_kv(self.encoded_nodes.clone().detach())

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        state_dict = {}

        if state.current_node is None:
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))
            probs = prob[:, :, None].expand(batch_size, pomo_size, self.problem_size).clone()
            probs.zero_()
            probs.scatter_(2, selected.unsqueeze(-1), 1)

            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_node)
            # state_embed = torch.zeros((batch_size, pomo_size, self.qkv_dim*self.head_num))
            state_dict['embed_node'] = encoded_first_node
            state_dict['ninf_mask'] = state.ninf_mask
        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            probs, q_last_concat = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem)
            state_dict['embed_node'] = encoded_last_node
            state_dict['ninf_mask'] = state.ninf_mask
            # state_embed = self.decoder.q_first + q_last_concat
            # state_embed = state_embed.reshape(batch_size, pomo_size, self.qkv_dim*self.head_num)
            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:

                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    # prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    batch_idx = state.BATCH_IDX.detach() if isinstance(state.BATCH_IDX, torch.Tensor) else state.BATCH_IDX
                    pomo_idx = state.POMO_IDX.detach() if isinstance(state.POMO_IDX, torch.Tensor) else state.POMO_IDX
                    selected = selected.detach() if isinstance(selected, torch.Tensor) else selected
                    prob = probs[batch_idx, pomo_idx, selected].clone().reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    if (prob > 1e-8).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None

        # self.decoder.q_first只取第一个pomo
        return selected, probs, prob, state_dict, self.decoder.q_first[:, :, 0, :]

def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data):
        # data.shape: (batch, problem, 2)

        embedded_input = self.embedding(data)
        # shape: (batch, problem, embedding)

        out = embedded_input
        for layer in self.layers:
            out = layer(out)

        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3
        # shape: (batch, problem, EMBEDDING_DIM)


########################################
# DECODER
########################################

class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_params['head_num']
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=head_num)
        # shape: (batch, n, head_num*qkv_dim)

    def forward(self, encoded_last_node, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        q = self.q_first + q_last
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs, q

########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed

def reshape_by_heads_steps(qkv, head_num):
    # q.shape: (steps, batch, head_num*key_dim)

    steps = qkv.size(0)
    batch_s = qkv.size(1)

    q_reshaped = qkv.reshape(steps, batch_s, head_num, -1)

    return q_reshaped

def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat

def multi_head_attention_steps(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (steps, batch, head_num, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (steps, batch, problem)

    steps = q.size(0)
    batch_s = q.size(1)
    head_num = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)
    score = torch.matmul(q.unsqueeze(3), k.transpose(2, 3)).squeeze()
    # shape: (steps, batch, head_num, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    
    # if rank2_ninf_mask is not None:
    #     score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, :, None, :].expand(steps, batch_s, head_num, input_s)

    score_scaled[-1, :, :, :] = -1e20
    weights = nn.Softmax(dim=3)(score_scaled)
    # ninf_mask of last state is fully masked, so the softmax output is nan, which is replaced by 0
    # shape: (steps, batch, head_num, problem)

    out = torch.matmul(weights.unsqueeze(3), v).squeeze()
    # shape: (steps, batch, head_num, key_dim)

    out_concat = out.reshape(steps, batch_s, head_num * key_dim)
    # shape: (steps, batch, head_num*key_dim)

    return out_concat

class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))

class CriticNetwork(torch.nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        # specifics of the network architecture
        self.network = torch.nn.Sequential(
            # torch.nn.Linear(env.observation_space.shape[0], n_nodes),
            torch.nn.Linear(128, 256), # 128 is the embedding dimension
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        ).float()
        # optimizer for the Critic Network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.01)



    # returns the state-value of a state, in numpy form: V(state)
    def predict(self, state):
        '''
        :param state: np.array of batched/single state
        :return: np.array of action probabilities
        '''
        if state.ndim < 2:
            values = self.network(torch.FloatTensor(state).unsqueeze(0).float())
        else:  # for state batch
            values = self.network(torch.FloatTensor(state))

        return values
    #enddef

    def update(self, states, targets):
        '''
        :param states: np.array of batched states
        :param targets: np.array of values
        :return: -- # performs 1 update on Critic Network
        '''
        pred_batch = self.network(states)
        loss = torch.nn.functional.smooth_l1_loss(pred_batch, targets.unsqueeze(-1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

# class RexploitNetwork(torch.nn.Module):
#     def __init__(self, **model_params):
#         super().__init__()
#         # specifics of network architecture
#         self.network = TSP_Decoder(**model_params)
#         # optimizer for the Actor Network
#         self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.01)

#         for param in self.network.parameters():
#             param.requires_grad = True

class RexploitNetwork(TSP_Decoder):
    def __init__(self, **model_params):
        super().__init__(**model_params)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, encoded_last_node, ninf_mask):
        # encoded_last_node.shape: (steps, batch, embedding)
        # ninf_mask.shape: (steps, batch, problem)
        # print(ninf_mask[0])
        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        q_last = reshape_by_heads_steps(self.Wq_last(encoded_last_node), head_num=head_num)
        # shape: (steps, batch, head_num, qkv_dim)

        q = self.q_first + q_last
        out_concat = multi_head_attention_steps(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (steps, batch, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (steps, batch, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out.unsqueeze(2), self.single_head_key).squeeze()
        # shape: (steps, batch, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (steps, batch, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask
        # shape: (steps, batch, problem)
        score_masked[-1, :, :] = -1e20
        
        probs = nn.Softmax(dim=2)(score_masked)
        
        # shape: (steps, batch, problem)
        return probs

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_params['head_num']
        self.k = reshape_by_heads(self.Wk(encoded_nodes).detach(), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes).detach(), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)