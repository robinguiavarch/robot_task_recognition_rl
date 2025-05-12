# agent1_patterns_chests_to_reach/approach2_temporal_window/agents/actor_lstm_agent.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple, Optional


class ActorLSTM(nn.Module):
    """
    π_θ(a | s, h)  –  policy-only network with a single LSTM layer.
    Le hidden-state n’est PAS réinitialisé automatiquement : le code
    d’entraînement doit appeler   h, c = model.init_hidden()  à chaque reset().
    """

    # --------------------------------------------------------------------- #
    # constructor
    def __init__(
        self,
        obs_dim:     int = 33,   # taille de ton vecteur encodé
        action_dim:  int = 8,    # 3 bits -> 8 actions
        embed_dim:   int = 64,   # petite projection avant LSTM
        lstm_hidden: int = 128,
    ):
        super().__init__()

        # 1) encoder |obs| -> embedding
        self.embed = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.ReLU(),
        )

        # 2) LSTM sur la séquence
        self.lstm = nn.LSTM(
            input_size   = embed_dim,
            hidden_size  = lstm_hidden,
            num_layers   = 1,
            batch_first  = True,
        )

        # 3) policy head  -> logits (pas de value head ici)
        self.policy_head = nn.Linear(lstm_hidden, action_dim)

        # 4) init orthogonale (optionnel mais stabilise)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)

    # ------------------------------------------------------------------ #
    # hidden-state helper
    def init_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
        c0 = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
        return h0, c0

    # ------------------------------------------------------------------ #
    # forward : accepte (B, obs) **ou** (B, T, obs)
    def forward(
        self,
        obs: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        single_step = obs.dim() == 2
        if single_step:
            obs = obs.unsqueeze(1)                        # (B,1,D)

        x = self.embed(obs)                               # (B,T,E)

        if hidden is None:
            hidden = self.init_hidden(obs.size(0), obs.device)

        lstm_out, next_hidden = self.lstm(x, hidden)      # (B,T,H)

        logits = self.policy_head(lstm_out)               # (B,T,A)

        if single_step:
            logits = logits.squeeze(1)                    # (B,A)

        return logits, next_hidden

    # ------------------------------------------------------------------ #
    # action helper (sampling ou greedy)
    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,                                # (1, obs_dim)
        hidden: Tuple[torch.Tensor, torch.Tensor] | None = None,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        logits, next_hidden = self.forward(obs, hidden)   # logits (1,A)
        probs = torch.softmax(logits, dim=-1)

        if deterministic:
            action_idx = torch.argmax(probs, dim=-1)      # (1,)
        else:
            action_idx = torch.multinomial(probs, 1)      # (1,1) -> (1,)
        log_prob = torch.log(probs.gather(1, action_idx.unsqueeze(-1)) + 1e-8)

        return action_idx.item(), log_prob.squeeze(), next_hidden
