from __future__ import annotations
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#                              Positional Encoding                            #
# --------------------------------------------------------------------------- #
class SinusoidalPositionalEncoding(nn.Module):
    """
    Classic sine / cosine positional encoding (Vaswani et al., 2017).

    * Shape in  : (B, T, d_model)
    * Shape out : (B, T, d_model)  with the encoding *added* in-place.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)          # (T,d)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # (T,1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10_000.0) / d_model)
        )                                           # (d/2,)

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)                        # (1,T,d)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B,T,d)
        x = x + self.pe[:, : x.size(1)]
        return x


# --------------------------------------------------------------------------- #
#                     O n - P o l i c y   T r a n s f o r m e r               #
# --------------------------------------------------------------------------- #
class OPTransformer(nn.Module):
    """
    OPTA – *On-Policy Transformer Actor* (policy-only, no critic).

    *Input*   : sequence of encoded observations  (B, T, obs_dim)
    *Output*  : logits for every timestep        (B, T, action_dim)

    For interactive use (env.loop) keep an internal buffer of the
    embeddings seen so far (cleared with ``reset_episode``).
    """

    # --------------------------------------------------------------------- #
    # constructor
    def __init__(
        self,
        obs_dim: int = 33,
        action_dim: int = 8,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        max_len: int = 2000,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 1) Observation embedding ϕ_obs
        self.embed = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.ReLU(),
        )

        # 2) Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)

        # 3) Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,          # (B,T,d) API
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # 4) Policy head → logits
        self.policy_head = nn.Linear(d_model, action_dim)

        # 5) (Optional) Orthogonal init for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)

        # ---- buffers for step-by-step interaction ----
        self._step_buffer: List[torch.Tensor] = []
        self.d_model = d_model

    # ------------------------------------------------------------------ #
    # episode helpers
    def reset_episode(self):
        """Call at every *env.reset()* to clear the internal sequence."""
        self._step_buffer.clear()

    # ------------------------------------------------------------------ #
    # forward : (B,T,obs_dim) or (T,obs_dim) for a single batch element
    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        obs_seq : tensor  (B, T, obs_dim)

        Returns
        -------
        logits  : tensor  (B, T, action_dim)
        """
        if obs_seq.dim() == 2:                       # (T,D) → (1,T,D)
            obs_seq = obs_seq.unsqueeze(0)

        x = self.embed(obs_seq)                      # (B,T,d)
        x = self.pos_enc(x)                          # add position
        h = self.transformer(x)                      # (B,T,d)
        logits = self.policy_head(h)                 # (B,T,A)
        return logits

    def act(
            self,
            obs: torch.Tensor,           # encoded obs_t   shape (obs_dim,)
            deterministic: bool = False,
            device: Optional[torch.device] = None,
        ) -> Tuple[int, torch.Tensor, torch.Tensor]:
            """
            Sample (or greedily pick) an action *with* gradient support.

            Args
            ----
            obs : encoded observation of current timestep (numpy→tensor OK)
            deterministic : if True → argmax, else categorical sampling
            device : optional torch.device to move `obs` onto

            Returns
            -------
            action_idx : int           chosen action ∈ [0..7]
            log_prob   : tensor()      log π(a_t | ·)  (scalar, requires_grad)
            probs      : tensor(8,)    action probabilities π(a_t | ·)
            """

            # allow passing numpy arrays
            if not torch.is_tensor(obs):
                obs = torch.tensor(obs, dtype=torch.float32)
            if device is not None:
                obs = obs.to(device)

            # 1) embed current obs and append to buffer
            e_t = self.embed(obs.unsqueeze(0))       # (1, d_model)
            self._step_buffer.append(e_t)            # list of (1, d_model)

            # 2) build sequence of embeddings so far
            seq = torch.stack(self._step_buffer, dim=1)  # (1, T, d_model)
            seq = self.pos_enc(seq)                      # add positional encodings

            # 3) transformer → last hidden
            h = self.transformer(seq)                    # (1, T, d_model)
            logits_t = self.policy_head(h[:, -1])        # (1, action_dim)

            # 4) build a distribution and sample/greedy
            probs = torch.softmax(logits_t, dim=-1).squeeze(0)  # (action_dim,)
            if deterministic:
                action_idx = torch.argmax(probs, dim=-1).item()
            else:
                action_idx = torch.multinomial(probs, num_samples=1).squeeze(0).item()

            # 5) compute log_prob *with* gradient
            log_prob = torch.log(probs[action_idx] + 1e-8)  # scalar tensor

            return action_idx, log_prob, probs


    @torch.no_grad()
    def act_eval(
        self,
        obs: torch.Tensor,
        deterministic: bool = True,
        device: Optional[torch.device] = None,
    ) -> int:
        """
        Same as act(), but wrapped in no_grad() and only returns the action index.
        """
        action_idx, _log_prob, _probs = self.act(obs, deterministic, device)
        return action_idx