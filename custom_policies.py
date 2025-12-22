import torch
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

CNN_SUFFIX = "_CNN"


def parse_algo_name(algo_name: str):
    normalized = algo_name.upper()
    if normalized.endswith(CNN_SUFFIX):
        return normalized[: -len(CNN_SUFFIX)], True
    return normalized, False


class ReactionDiffusionCnnExtractor(BaseFeaturesExtractor):
    """Custom features extractor for spatial reaction-diffusion grids."""

    def __init__(self, observation_space: spaces.Dict, embedding_dim: int = 32):
        super().__init__(observation_space, features_dim=1)
        variables_shape = observation_space["variables"].shape
        n_channels = variables_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv3d(n_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.cell_embedding = nn.Embedding(observation_space["cell_line"].n, embedding_dim)
        self._features_dim = 32 + embedding_dim

    def forward(self, observations):  # pylint: disable=arguments-differ
        variables = observations["variables"].float()
        if variables.dim() == 4:
            variables = variables.unsqueeze(0)
        cnn_out = self.cnn(variables)
        cnn_out = torch.flatten(cnn_out, start_dim=1)

        cell_indices = observations["cell_line"].long()
        if cell_indices.dim() == 0:
            cell_indices = cell_indices.unsqueeze(0)
        # Stable Baselines can provide the discrete observation with an extra
        # dimension (e.g., (batch, n_stack)), so collapse any trailing dims and
        # keep a single index per batch item to align with the CNN output.
        if cell_indices.dim() >= 2:
            cell_indices = cell_indices.reshape(cell_indices.shape[0], -1)[:, 0]
        embedded_cell = self.cell_embedding(cell_indices)
        if embedded_cell.dim() == 3:
            embedded_cell = embedded_cell.squeeze(1)
        return torch.cat([cnn_out, embedded_cell], dim=1)


class ReactionDiffusionCnnPolicy(MultiInputActorCriticPolicy):
    """Multi-input policy using the reaction-diffusion CNN extractor."""

    def __init__(self, *args, **kwargs):
        extractor_kwargs = kwargs.pop("features_extractor_kwargs", {}) or {}
        extractor_kwargs.setdefault("embedding_dim", 32)
        super().__init__(
            *args,
            features_extractor_class=ReactionDiffusionCnnExtractor,
            features_extractor_kwargs=extractor_kwargs,
            **kwargs,
        )


__all__ = [
    "CNN_SUFFIX",
    "ReactionDiffusionCnnExtractor",
    "ReactionDiffusionCnnPolicy",
    "parse_algo_name",
]
