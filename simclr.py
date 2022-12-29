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
        bz = x1.shape[0]

        h1 = self.encoder(x1)
        z1 = self.projector(h1.squeeze())  # (bz, project_dim)
        z1 = F.normalize(z1, p=2, dim=1)  # (bz, project_dim)

        h2 = self.encoder(x2)
        z2 = self.projector(h2.squeeze())  # (bz, project_dim)
        z2 = F.normalize(z2, p=2, dim=1)  # (bz, project_dim)

        similarity = torch.matmul(z1, z2.T) / self.temperature  # (bz, bz)
        assert list(similarity.shape) == [bz, bz]

        exp_similarity = similarity.exp()
        pos_similarity = torch.diag(similarity)  # (bz,)
        neg_similarity = exp_similarity.sum(dim=0) + exp_similarity.sum(dim=1) - 2 * pos_similarity.exp()  # (bz,)

        assert len(pos_similarity) == bz
        assert len(neg_similarity) == bz

        loss = - pos_similarity.mean() + neg_similarity.log().mean()

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
        model_path = str(dir_path / 'simclr_learner.pt')

        checkpoint = {
            'encoder': self.encoder.to(cpu_device).state_dict(),
            'projector': self.projector.to(cpu_device).state_dict()
        }

        torch.save(checkpoint, model_path)

        self.encoder.to(self.device)
        self.projector.to(self.device)

    def load(self, model_path):
        if not model_path.exist():
            raise FileNotFoundError()

        checkpoint = torch.load(str(model_path))

        self.encoder.load_state_dict(checkpoint['encoder'])
        self.projector.load_state_dict(checkpoint['projector'])

        self.encoder.to(self.device)
        self.projector.to(self.device)
