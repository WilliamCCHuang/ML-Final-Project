import torch
import torch.nn as nn
import torch.nn.functional as F


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
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

        
class SimCLR(nn.Module):

    def __init__(
        self,
        encoder,
        transform_1,
        transform_2,
        feature_dim=2048,
        project_dim=128,
        temperature=0.5
    ):
        super().__init__()

        if transform_1 is None:
            raise ValueError('Must assign `transform_1`')

        if transform_2 is None:
            raise ValueError('Must assign `transform_2`')

        self.device = next(encoder.parameters()).device

        self.feature_dim = feature_dim
        self.project_dim = project_dim

        self.transform_1 = transform_1
        self.transform_2 = transform_2
        self.temperature = temperature

        self.encoder = encoder
        self.projector = MLP(feature_dim, project_dim, project_dim).to(self.device)

    def _compute_loss(self, x1, x2):
        h1 = self.encoder(x1)
        z1 = self.projector(h1.squeeze())  # (bz, project_dim)
        z1 = F.normalize(z1, p=2, dim=1)  # (bz, project_dim)

        h2 = self.encoder(x2)
        z2 = self.projector(h2.squeeze())  # (bz, project_dim)
        z2 = F.normalize(z2, p=2, dim=1)  # (bz, project_dim)

        similarity = torch.matmul(z1, z2.T) / self.temperature  # (bz, bz)
        numerator = torch.diag(similarity)  # (bz,)
        denominator = torch.logcumsumexp(dim=0) + torch.logcumsumexp(dim=1)  # (bz,)

        # similarity = similarity.exp()
        # exp_pos_similarity = pos_similarity.exp()
        # neg_col_similarity = similarity.sum(dim=0) - exp_pos_similarity
        # neg_row_similarity = similarity.sum(dim=1) - exp_pos_similarity

        loss = - numerator.mean() + denominator.mean()

        return loss

    def forward(self, x):
        x1 = self.transform_1(x)
        x2 = self.transform_2(x)

        x1 = x1.to(self.device)
        x2 = x2.to(self.device)

        loss = self._compute_loss(x1, x2)

        return loss

    def get_representation(self, x):
        bz = x.shape[0]
        h = self.encoder(x)

        return h.view(bz, -1)

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
