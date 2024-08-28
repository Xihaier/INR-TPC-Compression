import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset, DataLoader


class WireLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, a=1.0, s=1.0):
        super().__init__()
        self.a = a
        self.s = s
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.a, np.sqrt(6 / self.in_features) / self.a)
        
    def forward(self, input):
        x = self.linear(input)
        return torch.sin(self.a * x) * torch.exp(-(self.s * x)**2)


class WIRERegressor(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True, a=1.0, s=1.0):
        super().__init__()
        self.net = []
        self.net.append(WireLayer(in_features, hidden_features, is_first=True, a=a, s=s))
        
        for i in range(hidden_layers-1):
            self.net.append(WireLayer(hidden_features, hidden_features, a=a, s=s))
        
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / a, np.sqrt(6 / hidden_features) / a)
                
            self.net.append(final_linear)
        else:
            self.net.append(WireLayer(hidden_features, out_features, a=a, s=s))
        
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


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


class WIRE:
    def __init__(self, params):
        self.params = params
        self.device = params["device"]
        self.model = WIRERegressor(
            in_features=params["in_features"],
            hidden_features=params["hidden_features"],
            hidden_layers=params["hidden_layers"],
            out_features=params["out_features"],
            outermost_linear=params.get("outermost_linear", True),
            a = params.get("a", 1),
            s = params.get("s", 1)
        ).to(self.device)
    
    def train_sparse(self, data, total_steps=1000, summary_interval=100, zero_fraction=0.1, refresh_interval=100):
        array_loader = self.create_loader(data)
        grid, array = next(iter(array_loader))
        grid, array = grid.squeeze().to(self.device), array.squeeze().to(self.device)
        train_coords, train_values = grid.reshape(-1, 3), array.reshape(-1, 1)
        
        # Identify non-zero values
        non_zero_indices = train_values.nonzero(as_tuple=True)[0]
        zero_indices = (train_values == 0).nonzero(as_tuple=True)[0]

        test_coords, test_values = grid.reshape(-1, 3), array.reshape(-1, 1)
        optim = torch.optim.Adam(lr=self.params["lr"], params=self.model.parameters())

        sampled_indices = None

        for step in range(1, total_steps + 1):
            # Sample a specified percentage of zero values every refresh_interval steps
            if step % refresh_interval == 1 or sampled_indices is None:
                num_zeros_to_sample = int(len(zero_indices) * zero_fraction)
                sampled_zero_indices = np.random.choice(zero_indices.cpu().numpy(), num_zeros_to_sample, replace=False)
                
                # Combine non-zero and sampled zero indices
                sampled_indices = torch.cat([non_zero_indices, torch.tensor(sampled_zero_indices, device=self.device)])
            
            # Filter the training coordinates and values
            current_train_coords = train_coords[sampled_indices]
            current_train_values = train_values[sampled_indices]
            
            self.model.train()
            optim.zero_grad()
            output = self.model(current_train_coords)
            train_loss = torch.nn.functional.mse_loss(output, current_train_values)
            train_loss.backward()
            optim.step()

            if not step % summary_interval:
                self.model.eval()
                with torch.no_grad():
                    prediction = self.model(test_coords)
                    test_loss = torch.nn.functional.mse_loss(prediction, test_values)
                    print(f"Step: {step}, Test MSE: {test_loss.item():.6f}")

        return test_loss.item()

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
            return self.model(coords.to(self.device))

    @staticmethod
    def create_loader(data):
        array_data = ArrayDataset(data)
        return DataLoader(array_data, batch_size=1)

    def get_compression_ratio(self, original_size):
        model_size = sum(p.numel() for p in self.model.parameters())
        return original_size / model_size