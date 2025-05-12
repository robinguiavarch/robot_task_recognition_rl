# transformer_dqn_agent.py

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from agent1_patterns_chests_to_reach.approach3_advanced_sequence_modeling.encoders.encoder_opta import encode_symbol_bg_fg, convert_index_to_action

from agent1_patterns_chests_to_reach.approach3_advanced_sequence_modeling.encoders.encoders_sliding_window import WINDOW_SIZE
from agent1_patterns_chests_to_reach.approach3_advanced_sequence_modeling.encoders.encoders_sliding_window import action_encoder

from agent1_patterns_chests_to_reach.approach3_advanced_sequence_modeling.encoders.encoders_sliding_window import (
    WINDOW_SIZE,
    sliding_window_encoder,
    reset_sliding_window,
    action_encoder,
    convert_index_to_action,
)

# --------------------------------------------------------------------------- #
#                       Positional Encoding (sin/cos)                         #
# --------------------------------------------------------------------------- #
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
#                        Transformer Q‐Network (per‐token embed)              #
# --------------------------------------------------------------------------- #
class TransformerDQN(nn.Module):
    def __init__(
        self,
        token_dim: int = 33,
        action_dim: int = 8,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        max_len: int = WINDOW_SIZE,
        dropout: float = 0.1,
    ):
        super().__init__()
        # 1) embed each 33‐D slot → d_model
        self.embed = nn.Linear(token_dim, d_model)
        # 2) positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        # 3) transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # 4) Q‐value head (from last time step)
        self.q_head = nn.Linear(d_model, action_dim)

        # orthogonal init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 33) → returns Q(s,a): (B, action_dim)
        """
        h = self.embed(x)              # → (B, T, d_model)
        h = self.pos_enc(h)            # add sinusoidal
        h = self.transformer(h)        # → (B, T, d_model)
        return self.q_head(h[:, -1])   # take last time step → (B, action_dim)


# --------------------------------------------------------------------------- #
#                        Full Agent with Double‐DQN                          #
# --------------------------------------------------------------------------- #
class TransformerDQNAgent:
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        device: str = "cpu",
        gamma: float = 0.99,
        lr: float = 1e-4,
        eps_start: float = 1.0,
        eps_end: float = 0.1,
        eps_decay: float = 0.995,
    ):
        self.device = torch.device(device)
        # Q network & target network
        self.q_net = TransformerDQN(token_dim=input_dim // WINDOW_SIZE,
                                    action_dim=action_dim).to(self.device)
        self.target_net = TransformerDQN(token_dim=input_dim // WINDOW_SIZE,
                                         action_dim=action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma

        # epsilon schedule
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def select_action(self, state: np.ndarray) -> int:
        """
        state: flat numpy array of shape (WINDOW_SIZE * token_dim,)
        returns action index [0..action_dim-1]
        """
        # reshape to (1, WINDOW_SIZE, token_dim)
        token_dim = state.shape[0] // WINDOW_SIZE
        s = (
            torch.tensor(state, dtype=torch.float32, device=self.device)
            .view(1, WINDOW_SIZE, token_dim)
        )
        if random.random() < self.eps:
            return random.randrange(self.q_net.q_head.out_features)
        with torch.no_grad():
            q = self.q_net(s)  # (1, action_dim)
        return int(q.argmax(dim=1).item())

    def update(self, batch: tuple) -> float:
        """
        batch: (states, actions, rewards, next_states, dones)
          states      : np.ndarray (B, WINDOW_SIZE*token_dim)
          actions     : np.ndarray (B,) int
          rewards     : np.ndarray (B,) float
          next_states : np.ndarray (B, WINDOW_SIZE*token_dim)
          dones       : np.ndarray (B,) bool
        """
        states, actions, rewards, next_states, dones = batch
        B = states.shape[0]
        token_dim = states.shape[1] // WINDOW_SIZE

        # to tensors & reshape
        s = (
            torch.tensor(states, dtype=torch.float32, device=self.device)
            .view(B, WINDOW_SIZE, token_dim)
        )
        a = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)  # (B,1)
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)  # (B,1)
        d = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)  # (B,1)
        s2 = (
            torch.tensor(next_states, dtype=torch.float32, device=self.device)
            .view(B, WINDOW_SIZE, token_dim)
        )

        # current Q(s,a)
        q_vals = self.q_net(s).gather(1, a)  # (B,1)

        # Double‐DQN target:
        with torch.no_grad():
            # argmax action from main net
            next_q_main = self.q_net(s2)
            next_a = next_q_main.argmax(dim=1, keepdim=True)  # (B,1)
            # evaluate on target net
            next_q_tgt = self.target_net(s2).gather(1, next_a)  # (B,1)
            target = r + self.gamma * next_q_tgt * (1 - d)

        loss = F.mse_loss(q_vals, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # decay ε
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        return loss.item()

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
