import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset, DataLoader


class FourierLayer(nn.Module):
    def __init__(self, in_features, out_features, weight_scale):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.data *= weight_scale  # gamma
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        return torch.sin(self.linear(x))


class FourierNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_layers=3, input_scale=256.0, weight_scale=1.0, bias=True, output_act=False):
        super().__init__()
        self.filters = nn.ModuleList(
            [FourierLayer(in_size, hidden_size, input_scale / np.sqrt(n_layers + 1)) for _ in range(n_layers + 1)]
        )
        self.linear = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias) for _ in range(n_layers)]
        )
        self.output_linear = nn.Linear(hidden_size, out_size)
        self.output_act = output_act

        for lin in self.linear:
            lin.weight.data.uniform_(
                -np.sqrt(weight_scale / hidden_size),
                np.sqrt(weight_scale / hidden_size),
            )

    def forward(self, x):
        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.linear[i - 1](out)
        out = self.output_linear(out)
        if self.output_act:
            out = torch.sin(out)
        return out


class ArrayDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.arr = data

    def __getitem__(self, idx):
        tensor_arr = torch.from_numpy(self.arr).float()
        h_axis = torch.linspace(0, 1, steps=self.arr.shape[0])
        w_axis = torch.linspace(0, 1, steps=self.arr.shape[1])
        d_axis = torch.linspace(0, 1, steps=self.arr.shape[2])
        grid = torch.stack(torch.meshgrid(h_axis, w_axis, d_axis, indexing='ij'), dim=-1)
        return grid, tensor_arr

    def __len__(self):
        return 1


class FFNet:
    def __init__(self, params):
        self.params = params
        self.device = params["device"]
        self.model = FourierNet(
            in_size=params["in_features"],
            hidden_size=params["hidden_features"],
            out_size=params["out_features"],
            n_layers=params["hidden_layers"],
            input_scale=params.get("input_scale", 256.0),
            weight_scale=params.get("weight_scale", 1.0),
            bias=True,
            output_act=False
        ).to(self.device)

    def train(self, data, total_steps=1000, summary_interval=100):
        best_mse_transformed = float('inf')
        
        array_loader = self.create_loader(data)
        grid, array = next(iter(array_loader))
        grid, array = grid.squeeze().to(self.device), array.squeeze().to(self.device)
        train_coords, train_values = grid.reshape(-1, 3), array.reshape(-1, 1)
        test_coords, test_values = grid.reshape(-1, 3), array.reshape(-1, 1)

        optim = torch.optim.Adam(lr=self.params["lr"], params=self.model.parameters())

        for step in range(1, total_steps + 1):
            
            self.model.train()
            optim.zero_grad()
            output = self.model(train_coords)
            train_loss = torch.nn.functional.mse_loss(output, train_values)
            train_loss.backward()
            optim.step()

            if not step % summary_interval:
                self.model.eval()
                with torch.no_grad():
                    prediction = self.model(test_coords)
                    test_loss = torch.nn.functional.mse_loss(prediction, test_values)
                    if test_loss.item() < best_mse_transformed:
                        best_mse_transformed = test_loss.item()
                        self.model.best_model_state = self.model.state_dict()
                    print(f"Step: {step}, Test MSE: {test_loss.item():.6f}")

        return best_mse_transformed

    def predict(self, coords):
        self.model.eval()
        # Load the best model state
        self.model.load_state_dict(self.model.best_model_state)
        with torch.no_grad():
            return self.model(coords)

    @staticmethod
    def create_loader(data):
        array_data = ArrayDataset(data)
        return DataLoader(array_data, batch_size=1)

    def get_compression_ratio(self, original_size):
        model_size = sum(p.numel() for p in self.model.parameters())
        return original_size / model_size