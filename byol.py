import copy
import math

import torch
import torch.nn as nn

from losses import BYOLLoss


class MLP(nn.Module):
    """MLP for projector and predictor
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class BYOL(nn.Module):

    def __init__(
        self,
        encoder,
        feature_dim=2048,
        projection_dim=256,
        projection_hidden_dim=4096,
        tau_base=0.996,
        total_training_steps=None
    ):
        super().__init__()

        if total_training_steps is None:
            raise ValueError('Must assign `total_training_steps`')

        self.device = next(encoder.parameters()).device

        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        self.projection_hidden_dim = projection_hidden_dim
        self.tau_base = tau_base
        self.total_training_steps = total_training_steps

        self.online_encoder = encoder
        self.online_projector = MLP(feature_dim, projection_hidden_dim, projection_dim).to(self.device)
        self.online_predictor = MLP(projection_dim, projection_hidden_dim, projection_dim).to(self.device)

        self.target_encoder = copy.deepcopy(self.online_encoder).to(self.device)
        self.target_projector = copy.deepcopy(self.online_projector).to(self.device)

        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

        self.loss_fn = BYOLLoss()

    def trainable_parameters(self):
        params = list(self.online_encoder.parameters())
        params += list(self.online_projector.parameters())
        params += list(self.online_predictor.parameters())
        
        return params

    def _compute_loss(self, x1, x2):
        online_repr_1 = self.online_encoder(x1)  # (bz, hidden_dim, 1, 1)
        online_repr_2 = self.online_encoder(x2)  # (bz, hidden_dim, 1, 1)
        online_proj_1 = self.online_projector(online_repr_1.squeeze())
        online_proj_2 = self.online_projector(online_repr_2.squeeze())
        online_pred_1 = self.online_predictor(online_proj_1)
        online_pred_2 = self.online_predictor(online_proj_2)

        with torch.no_grad():
            target_repr_1 = self.target_encoder(x1)  # (bz, hidden_dim, 1, 1)
            target_repr_2 = self.target_encoder(x2)  # (bz, hidden_dim, 1, 1)
            target_proj_1 = self.target_projector(target_repr_1.squeeze())
            target_proj_2 = self.target_projector(target_repr_2.squeeze())

            target_proj_1.detach_()
            target_proj_2.detach_()

        loss_1 = self.loss_fn(online_pred_1, target_proj_2)
        loss_2 = self.loss_fn(online_pred_2, target_proj_1)
        loss = (loss_1 + loss_2).mean()
        
        return loss

    def forward(self, x1, x2):
        loss = self._compute_loss(x1, x2)

        return loss

    def _compute_tau(self, current_training_steps):
        assert current_training_steps <= self.total_training_steps
        return 1 - (1 - self.tau_base) * (math.cos(math.pi * current_training_steps / self.total_training_steps) + 1) / 2

    @torch.no_grad()
    def update_target_network(self, current_training_steps):
        tau = self._compute_tau(current_training_steps)

        for online_param, target_param in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            new_weight = online_param.data
            old_weight = target_param.data

            target_param.data = tau * old_weight + (1 - tau) * new_weight

        for online_param, target_param in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            new_weight = online_param.data
            old_weight = target_param.data

            target_param.data = tau * old_weight + (1 - tau) * new_weight

    def get_representation(self, x):
        bz = x.shape[0]
        repr = self.online_encoder(x)

        return repr.view(bz, -1)

    def save(self, dir_path):
        cpu_device = torch.device('cpu')
        model_path = str(dir_path / 'byol_learner.pt')

        checkpoint = {
            'online_encoder': self.online_encoder.to(cpu_device).state_dict(),
            'online_projector': self.online_projector.to(cpu_device).state_dict(),
            'online_predictor': self.online_predictor.to(cpu_device).state_dict()
        }

        torch.save(checkpoint, model_path)

        self.online_encoder.to(self.device)
        self.online_projector.to(self.device)
        self.online_predictor.to(self.device)

    def load(self, model_path):
        if not model_path.exist():
            raise FileNotFoundError()

        checkpoint = torch.load(str(model_path))

        self.online_encoder.load_state_dict(checkpoint['online_encoder'])
        self.online_projector.load_state_dict(checkpoint['online_projector'])
        self.online_predictor.load_state_dict(checkpoint['online_predictor'])

        self.online_encoder.to(self.device)
        self.online_projector.to(self.device)
        self.online_predictor.to(self.device)
