import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from einops import rearrange, repeat
from matplotlib import pyplot as plt


data_path = 'data.mat'
epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dim = 3


def main(time_steps):

    def rollout(trajectory, time_steps):
        x = []
        y = []
        for i in range(trajectory.shape[0] - time_steps):
            x.append(trajectory[i:i+time_steps])
            y.append(trajectory[i+time_steps])
        return np.array(x), np.array(y)

    def get_loader(trajectories, batch_size, time_steps, shuffle):
        xs = []
        ys = []
        for trajectory in rearrange(trajectories, "d t b -> b t d"):
            x, y = rollout(trajectory, time_steps)
            xs.append(x)
            ys.append(y)

        xs = np.concatenate(xs)
        ys = np.concatenate(ys)
        dataset = TensorDataset(torch.from_numpy(xs).float(), torch.from_numpy(ys).float())
        # print(xs.shape)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader

    data = scipy.io.loadmat(data_path)
    trajectory_train, trajectory_test = data['trajectory_train'], data['trajectory_test']

    train_loader = get_loader(trajectory_train, batch_size=32, time_steps=time_steps, shuffle=True)
    test_loader = get_loader(trajectory_test, batch_size=32, time_steps=time_steps, shuffle=False)

    model = RK4()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch_idx in tqdm(range(epochs)):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            # x=[b t d] -> y=[b d]
            # print(x.shape)
            y_pred = model(x)
            # print(x.shape, y_pred.shape, y.shape)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch_idx} train: {train_loss}")

        model.eval()
        test_loss = 0
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            test_loss += loss.item()
        print(f"Epoch {epoch_idx} eval: {test_loss}")

        if epoch_idx % 10 == 9:
            max_time = 1000
            predictions = []
            x = rearrange(torch.from_numpy(trajectory_train[:,0:time_steps,:]), "d t b -> b t d").float().to(device)
            with torch.no_grad():
                for i in range(max_time):
                    y_pred = model(x)
                    # print(y_pred.shape, x.shape)
                    predictions.append(y_pred.detach().cpu().numpy())
                    x = torch.cat([x[:,1:,:], y_pred.unsqueeze(1)], dim=1)

            predictions = np.stack(predictions, axis=1)
            draw_compare(trajectory_train, predictions, epoch_idx)


def draw_compare(trajectory_test, predictions, epoch_idx):
    trajectory_test = rearrange(trajectory_test, "d t b -> b t d")

    for bi, (test_data, pred_data) in enumerate(zip(trajectory_test, predictions)):

        plt.figure(figsize=(12, 8))
        time = np.arange(trajectory_test.shape[1])

        for pi, l in zip(range(3), ['X', 'Y', 'Z']):
            plt.subplot(3, 1, pi+1)
            plt.plot(time, test_data[:, pi], 'b', label='ground')
            plt.plot(time, pred_data[:, pi], 'r', label='pred')
            plt.title(l + ' sequence')
            plt.ylabel(l)
            if pi == 2:
                plt.xlabel('Time')
            plt.legend()

        plt.tight_layout()
        plt.savefig(f'pics/{bi}-{epoch_idx}.png')

class Hidden(nn.Module):

    def __init__(self, hidden_size=100):
        super(Hidden, self).__init__()
        self.activation = nn.Sigmoid()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs):
        return self.activation(self.norm(self.linear(inputs)) + inputs)


class Model(nn.Module):
    def __init__(self, input_size=dim, output_size=1, hidden_size=100, heads=2, layers=2):
        super(Model, self).__init__()
        self.linear_in = nn.Linear(input_size, hidden_size)
        get_head = lambda: nn.Sequential(
            *[Hidden(hidden_size) for _ in range(layers)]
        )
        self.heads = nn.ModuleList([get_head() for _ in range(heads)])
        self.linear_out=  nn.Linear(hidden_size * heads, output_size)

    def forward(self, inputs):
        hidden = self.linear_in(inputs)
        hiddens = torch.concat([head(hidden) for head in self.heads], dim=1)
        return self.linear_out(hiddens)


class RK4(nn.Module):
    def __init__(self, delta=0.1):
        super(RK4, self).__init__()
        self.models = nn.ModuleList([Model() for _ in range(dim)])
        self.delta = delta

    def forward(self, inputs):
        """inputs: [b 1 d]"""
        results = []
        for idx, model in enumerate(self.models):
            results.append(self.forward_one(model, inputs.clone(), inputs[:,:,idx]))
        results = torch.concat(results, dim=1)
        return results

    def forward_one(self, model, inputs, x):
        """inputs: [b 1 d]"""
        comp = dict(size=(inputs.shape[0], 2), device=device)
        data = inputs.squeeze(1)
        k1 = model(data)  # [b d] -> [b]
        data = data + self.delta / 2 * torch.concat([torch.ones(**comp), k1], dim=1)
        k2 = model(data)
        data = data + torch.concat([self.delta / 2 * (k2 - k1), torch.zeros(**comp)], dim=1)
        k3 = model(data)
        data = data + self.delta / 2 * torch.concat([k3, torch.ones(**comp)], dim=1)
        k4 = model(data)
        return x + self.delta / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


if __name__ == '__main__':

    main(1)
