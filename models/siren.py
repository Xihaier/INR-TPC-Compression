import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset, DataLoader


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class SIRENRegressor(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear,first_omega_0,hidden_omega_0):
        super().__init__()
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))
        
        for i in range(hidden_layers-1):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))
        
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
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


def entropy_sampling(xyz, points, nBins, npoint):
    """
    Input:
        xyz: (batch_size, number of data points, 3)
        points: (batch_size, number of data points, number of features)
        nBins: number of bins for histogram
        npoint: number of samples
    Return:
        centroids: sampled pointcloud points, [B, npoint]
    """
    sampled_xyz_batches = []
    sampled_feats_batches = []
    for bidx in range(points.shape[0]):
        # build histogram and sort by number of elements
        minval = torch.min(points[bidx][...,0])
        maxval = torch.max(points[bidx][...,0])
        hist = torch.histc(points[bidx][...,0], bins=nBins, min=minval, max=maxval)
        bin_min = minval
        bin_width = (maxval - minval) / nBins
        suffle_idx = torch.randperm(points[bidx].size()[0])
        shuffle_xyz = xyz[bidx][suffle_idx]
        shuffle_feat = points[bidx][suffle_idx]
        sort_hist = []
        for i in range(len(hist)):
            sort_hist.append(torch.Tensor([hist[i], minval + bin_width*i, minval + bin_width*(i+1)]))
        sort_hist = torch.stack(sort_hist)
        hist, ind = sort_hist.sort(0)
        sort_hist = sort_hist[ind[...,0]]

        # compute the number of points for each bin
        sampled_xyz = []
        sampled_feats = []
        sampled_hist_list = [0 for i in range(nBins)]
        sampled_hist = []
        curSamples = npoint
        curBins = nBins
        for i in range(nBins):
            npoint_bin = curSamples // curBins
            npoint_bin = min(npoint_bin, sort_hist[i][0].to(int))
            sampled_hist_list[i] = npoint_bin
            curSamples -= npoint_bin
            curBins -= 1
            sampled_hist.append(torch.Tensor([sort_hist[i][1], sort_hist[i][2], sampled_hist_list[i]]))
        sampled_hist = torch.stack(sampled_hist)
        hist, ind = sampled_hist.sort(0)
        sort_hist = sampled_hist[ind[...,0]]
        sampled_hist_list = sort_hist[...,2]
        
        # select points
        for i in range(shuffle_xyz.shape[0]):
            idx = min(torch.floor((shuffle_feat[i][0] - bin_min)/bin_width).to(int), nBins-1)
            if sampled_hist_list[idx] > 0:
                sampled_xyz.append(shuffle_xyz[i])
                sampled_feats.append(shuffle_feat[i])
                sampled_hist_list[idx] -= 1
            if torch.count_nonzero(sampled_hist_list) == 0:
                break
        sampled_xyz = torch.stack(sampled_xyz)
        sampled_feats = torch.stack(sampled_feats)
        sampled_xyz_batches.append(sampled_xyz)
        sampled_feats_batches.append(sampled_feats)
    sampled_xyz_batches = torch.stack(sampled_xyz_batches)
    sampled_feats_batches = torch.stack(sampled_feats_batches)
    return sampled_xyz_batches, sampled_feats_batches


class SIREN:
    def __init__(self, params):
        self.params = params
        self.device = params["device"]
        self.model = SIRENRegressor(
            in_features=params["in_features"],
            hidden_features=params["hidden_features"],
            hidden_layers=params["hidden_layers"],
            out_features=params["out_features"],
            outermost_linear=params.get("outermost_linear", True),
            first_omega_0=params["first_omega_zero"],
            hidden_omega_0=params["hidden_omega_zero"]
        ).to(self.device)

    def calculate_initial_weights(self,data):
        data_tensor = torch.from_numpy(data).float()
        weights = torch.where(data_tensor != 0, torch.abs(data_tensor), torch.tensor(0.1))
        return weights / weights.sum()
        
    def sample_data(self,data, weights, num_samples):
        # Flatten data and weights for sampling
        data_flat = data.view(-1)
        weights_flat = weights.view(-1)
    
        # Sample indices based on weights
        indices = torch.multinomial(weights_flat, num_samples, replacement=True)
    
        # Retrieve sampled data points
        sampled_data = data_flat[indices]
        return sampled_data, indices
    
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

        for step in range(1, total_steps + 1):
            # Sample a specified percentage of zero values every refresh_interval steps
            num_zeros_to_sample = int(len(zero_indices) * zero_fraction)
            sampled_zero_indices = torch.multinomial(torch.ones(len(zero_indices), device=self.device), num_zeros_to_sample, replacement=False)
            sampled_zero_indices = zero_indices[sampled_zero_indices]
                
            # Combine non-zero and sampled zero indices
            sampled_indices = torch.cat([non_zero_indices, sampled_zero_indices])
        
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

    def train_entropy(self, data, total_steps=1000, summary_interval=100, nBins=10, fraction=0.1):
        best_mse_transformed = float('inf')

        array_loader = self.create_loader(data)
        grid, array = next(iter(array_loader))
        grid, array = grid.squeeze().to(self.device), array.squeeze().to(self.device)
        train_coords, train_values = grid.reshape(-1, 3), array.reshape(-1, 1)
        test_coords, test_values = grid.reshape(-1, 3), array.reshape(-1, 1)

        optim = torch.optim.Adam(lr=self.params["lr"], params=self.model.parameters())

        npoint = int(train_coords.shape[0] * fraction)

        for step in range(1, total_steps + 1):
            print(step)
            # Apply entropy sampling
            sampled_coords, sampled_values = entropy_sampling(
                train_coords.unsqueeze(0),
                train_values.unsqueeze(0),
                nBins,
                npoint
            )
            sampled_coords = sampled_coords.squeeze(0)
            sampled_values = sampled_values.squeeze(0)

            self.model.train()
            optim.zero_grad()
            output = self.model(sampled_coords)
            train_loss = torch.nn.functional.mse_loss(output, sampled_values)
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

        return test_loss.item()

    def train(self, data, total_steps=1000, summary_interval=2):
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

    def train_importance(self, data, total_steps=1000, summary_interval=100, fraction=0.1):
        best_mse_transformed = float('inf')

        array_loader = self.create_loader(data)
        grid, array = next(iter(array_loader))
        grid, array = grid.squeeze().to(self.device), array.squeeze().to(self.device)
        train_coords, train_values = grid.reshape(-1, 3), array.reshape(-1, 1)
        
        # Calculate initial weights for importance sampling
        weights = self.calculate_initial_weights(array.cpu().numpy())
        weights = weights.to(self.device)
        
        test_coords, test_values = grid.reshape(-1, 3), array.reshape(-1, 1)
        optim = torch.optim.Adam(lr=self.params["lr"], params=self.model.parameters())
        
        for step in range(1, total_steps + 1):
            # Sample data points based on importance sampling
            num_samples = int(train_coords.shape[0] * fraction)
            _, sampled_indices = self.sample_data(train_values, weights, num_samples)
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
                    if test_loss.item() < best_mse_transformed:
                        best_mse_transformed = test_loss.item()
                        self.model.best_model_state = self.model.state_dict()
                    print(f"Step: {step}, Test MSE: {test_loss.item():.6f}")

        return test_loss.item()

    def train_random(self, data, total_steps=1000, summary_interval=100, fraction=0.1):
        best_mse_transformed = float('inf')

        array_loader = self.create_loader(data)
        grid, array = next(iter(array_loader))
        grid, array = grid.squeeze().to(self.device), array.squeeze().to(self.device)
        train_coords, train_values = grid.reshape(-1, 3), array.reshape(-1, 1)
    
        # Create uniform weights for random sampling
        weights = torch.ones(train_values.shape[0], device=self.device)
    
        test_coords, test_values = grid.reshape(-1, 3), array.reshape(-1, 1)
        optim = torch.optim.Adam(lr=self.params["lr"], params=self.model.parameters())
    
        for step in range(1, total_steps + 1):
            # Sample data points based on uniform weights
            num_samples = int(train_coords.shape[0] * fraction)
            sampled_indices = torch.multinomial(weights, num_samples, replacement=False)
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
                    if test_loss.item() < best_mse_transformed:
                        best_mse_transformed = test_loss.item()
                        self.model.best_model_state = self.model.state_dict()
                    print(f"Step: {step}, Test MSE: {test_loss.item():.6f}")

        return test_loss.item()

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