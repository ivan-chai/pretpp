import torch
from torch import Tensor
from typing import Tuple, Optional

from hotpp.data import PaddedBatch


class BidirGRU(torch.nn.GRU):
    """Bidirectional GRU interface."""
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self._hidden_size = hidden_size

    @property
    def output_size(self):
        return 2 * self._hidden_size

    @property
    def init_state(self):
        p = next(iter(self.parameters()))
        return torch.zeros(2, self.hidden_size, dtype=p.dtype, device=p.device)  # (2, D).

    def forward(self, x: PaddedBatch, time_deltas: PaddedBatch,
                states: Optional[Tensor]=None, return_full_states=False) -> Tuple[PaddedBatch, Tensor]:
        """Apply RNN.

        Args:
            x: Batch with shape (B, L, D).
            time_deltas (unused): Relative inputs timestamps.
            states: Initial states with shape (2 * N, B, D), where N is the number of layers.
            return_full_states: Whether to return full states with shape (2 * N, B, L, D)
                or only output states with shape (2 * N, B, D).

        Returns:
            Outputs with shape (B, L, D) and states with shape (N, B, D) or (N, B, L, D), where
            N is the number of layers.
        """
        empty_mask = x.seq_lens == 0  # (B).
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x.payload, x.seq_lens.clip(min=1).cpu(),
                                                           batch_first=True, enforce_sorted=False)
        outputs_packed, _ = super().forward(x_packed, states)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs_packed, total_length=x.payload.shape[1], batch_first=True)  # (B, L, 2 * D).
        if return_full_states:
            if self.num_layers == 1:
                # In GRU output and states are equal.
                outputs_fw, outputs_bw = outputs.split(self._hidden_size, dim=-1)  # 2 x (B, L, D).
                states = torch.stack([outputs_fw, outputs_bw])  # (2, B, L, D).
            else:
                raise NotImplementedError("Multilayer GRU states")
        else:
            outputs_last = outputs.take_along_dim((x.seq_lens - 1).clip(min=0)[:, None, None], 1).squeeze(1)  # (B, 2 * D).
            outputs_fw, outputs_bw = outputs_last.split(self._hidden_size, dim=-1)  # 2 x (B, D).
            states = torch.stack([outputs_fw, outputs_bw])  # (2, B, D).
        outputs = PaddedBatch(outputs, x.seq_lens)
        return outputs, states

    def interpolate(self, states: Tensor, time_deltas: PaddedBatch) -> PaddedBatch:
        """Compute model outputs in continuous time.

        Args:
            states: Model states with shape (2 * N, B, L, D), where N is the number of layers.
            time_deltas: Relative timestamps with shape (B, L, S), where S is a sample size.

        Returns:
            Outputs with shape (B, L, S, D).
        """
        # GRU output is constant between events.
        # Forward network predicts last seen event.
        # Backward network predicts the next event.
        s = time_deltas.payload.shape[2]
        _, b, l, d = states.shape
        outputs = states.reshape(2, self.num_layers, b, l, d)[:, -1]  # (2, B, L, D).
        forward_states = outputs[0]  # (B, L, D).
        backward_states = outputs[1]  # (B, L, D).
        backward_init_states = self.init_state[1][None, None].repeat(b, 1, 1)  # (B, 1, D).
        backward_states = torch.cat([backward_states[:, 1:], backward_init_states], 1)  # (B, L, D).
        outputs = torch.cat([forward_states, backward_states], -1)  # (B, L, 2 * D).
        outputs = outputs.unsqueeze(2).repeat(1, 1, s, 1)  # (B, L, S, 2 * D).
        return PaddedBatch(outputs, time_deltas.seq_lens)
