"""Simple recurrent neural networks: Elman, Jordan and multi-recurrent.

All three share one cell. They differ only in what is fed back into the
hidden layer, which keeps the architectural comparison clean:

    Elman     hidden state h(t-1)                      (context layer)
    Jordan    network output o(t-1)                    (state layer)
    MRNN      both, each through several memory banks with different
              decay rates, so the network sees memory at several timescales
"""
from __future__ import annotations

import torch
import torch.nn as nn

FEEDBACK = ("hidden", "output", "both")


class SimpleRNN(nn.Module):
    """One hidden layer with recurrent feedback, one linear output unit.

    Parameters
    ----------
    n_inputs   : number of exogenous inputs per time step (1 for univariate)
    n_hidden   : hidden units
    n_outputs  : output units (1 for one-step-ahead forecasting)
    feedback   : "hidden" (Elman), "output" (Jordan) or "both" (MRNN)
    decays     : memory-bank decay rates, MRNN only. 0.0 is a plain one-step
                 copy; larger values hold an exponentially decaying trace.
    """

    def __init__(self, n_inputs: int = 1, n_hidden: int = 8, n_outputs: int = 1,
                 feedback: str = "hidden", decays=(0.0, 0.3, 0.6, 0.9)):
        super().__init__()
        if feedback not in FEEDBACK:
            raise ValueError(f"feedback must be one of {FEEDBACK}")
        self.feedback = feedback
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.decays = tuple(decays) if feedback == "both" else (0.0,)
        self.n_banks = len(self.decays)

        if feedback == "hidden":
            n_context = n_hidden
        elif feedback == "output":
            n_context = n_outputs
        else:
            n_context = self.n_banks * (n_hidden + n_outputs)

        self.hidden = nn.Linear(n_inputs + n_context, n_hidden)
        self.act = nn.Tanh()
        self.out = nn.Linear(n_hidden, n_outputs)

    def init_state(self, batch: int, device, dtype):
        z = lambda k: torch.zeros(batch, k, device=device, dtype=dtype)
        return {
            "h": z(self.n_hidden),
            "o": z(self.n_outputs),
            "mh": [z(self.n_hidden) for _ in self.decays],
            "mo": [z(self.n_outputs) for _ in self.decays],
        }

    def step(self, x_t, state):
        """One time step. x_t: (batch, n_inputs)."""
        if self.feedback == "hidden":
            context = state["h"]
        elif self.feedback == "output":
            context = state["o"]
        else:
            context = torch.cat(state["mh"] + state["mo"], dim=1)

        h = self.act(self.hidden(torch.cat([x_t, context], dim=1)))
        o = self.out(h)

        new = {"h": h, "o": o}
        if self.feedback == "both":
            # m(t) = a * m(t-1) + (1 - a) * signal(t)
            new["mh"] = [a * m + (1 - a) * h for a, m in zip(self.decays, state["mh"])]
            new["mo"] = [a * m + (1 - a) * o for a, m in zip(self.decays, state["mo"])]
        else:
            new["mh"], new["mo"] = state["mh"], state["mo"]
        return o, new

    def forward(self, x):
        """x: (batch, seq_len, n_inputs) -> output at the final step."""
        state = self.init_state(x.shape[0], x.device, x.dtype)
        o = None
        for t in range(x.shape[1]):
            o, state = self.step(x[:, t, :], state)
        return o


def build_model(name: str, n_hidden: int, n_inputs: int = 1, **kw) -> SimpleRNN:
    feedback = {"elman": "hidden", "jordan": "output", "mrnn": "both"}[name]
    return SimpleRNN(n_inputs=n_inputs, n_hidden=n_hidden, feedback=feedback, **kw)


MODELS = ("elman", "jordan", "mrnn")