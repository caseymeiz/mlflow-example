import numpy as np
import os
import tempfile
import matplotlib.pyplot as plt
import mlflow
from dataclasses import dataclass
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    config = Config(lr=args.lr)
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Classifier")
    with mlflow.start_run(run_name="MLP"):
        X, y= create_data(config)
        fig = plot(X, y)
        save_fig(fig, "ground_truth.png")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = MLP()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        train(model, loss_fn, optimizer, X_train, y_train)
        test(model, X_test, y_test)


def train(model, loss_fn, optimizer, X_train_t, y_train_t):
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        optimizer.step()
        mlflow.log_metric("loss", loss, step=epoch)

def test(model, X_test_t, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        preds = torch.argmax(logits, dim=1).numpy()
    acc = accuracy_score(y_test, preds)
    mlflow.log_metric("accuracy", acc)

@dataclass
class Config:
    mean_a: tuple = (0, 0)
    mean_b: tuple = (2, 2)
    cov: tuple = ((1, 0), (0, 1))
    samples_per_class: int = 100
    lr: float = 0.001

def create_data(config):
    class_a = np.random.multivariate_normal(config.mean_a, config.cov, size=config.samples_per_class)
    class_b = np.random.multivariate_normal(config.mean_b, config.cov, size=config.samples_per_class)
    data_a = np.hstack((class_a, np.zeros((class_a.shape[0], 1))))
    data_b = np.hstack((class_b, np.ones((class_b.shape[0], 1))))
    data = np.vstack((data_a, data_b))
    X = torch.tensor(data[:, :2], dtype=torch.float32)
    y = torch.tensor(data[:, 2], dtype=torch.long)
    return X, y

class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=32, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def plot(X, y):
    X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
    y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
    class0 = X_np[y_np == 0]
    class1 = X_np[y_np == 1]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(class0[:, 0], class0[:, 1], label='Class 0', alpha=0.7)
    ax.scatter(class1[:, 0], class1[:, 1], label='Class 1', alpha=0.7)
    ax.legend()
    return fig

def save_fig(fig, name):
    with tempfile.TemporaryDirectory() as tmpdir:
        fig_path = os.path.join(tmpdir, name)
        fig.savefig(fig_path)
        mlflow.log_artifact(fig_path, artifact_path="plots")
    plt.close(fig)

if __name__ == "__main__":
    main()
