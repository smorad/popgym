from typing import List, Tuple

import gym
import torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from popgym.baselines.models.s4d import S4 as S4Model
from popgym.baselines.ray_models.base_model import BaseModel


class S4D(BaseModel):
    """A model for traditional RNNs. This supports LSTMs, Elman RNNs, and GRUs"""

    MODEL_CONFIG = {
        "embedding": None,
    }

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **custom_model_kwargs,
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.h = round(self.cfg["hidden_size"] ** 0.5 / 2) * 2
        self.map = torch.nn.Linear(self.cfg["preprocessor_output_size"], self.h)
        self.unmap = torch.nn.Linear(self.h, self.cfg["hidden_size"])
        self.core = S4Model(
            d_model=self.h,
            d_state=self.h,
            mode="diag",
            measure="diag-lin",
            bidirectional=False,
            disc="zoh",
            real_type="exp",
            transposed=False,
        )
        self.did_setup = False
        # Required to initialize some parts of S4

    def initial_state(self) -> List[TensorType]:
        state = self.core.default_state(1).squeeze(0)
        # Ray cannot handle complex floats
        real, imag = state.real, state.imag
        # For each batch, each tensor has shape [1, self.h, self.h]
        return [real, imag]

    def forward_memory(
        self,
        z: TensorType,
        state: List[TensorType],
        t_starts: TensorType,
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:

        # Calling setup_step in init causes deepcopy to fail
        if not self.did_setup:
            self.core.setup_step()
            self.did_setup = True
        real, imag = state
        memory = torch.complex(real, imag)
        z = self.map(z)
        # Inference mode
        if torch.torch.all(seq_lens == 1):
            z, memory = self.core.step(z.squeeze(1), memory)
            z = z.unsqueeze(1)
        # Train mode
        else:
            z, _ = self.core(z)

        z = self.unmap(z)

        return z, [memory.real, memory.imag]