import numpy as np  # manipulate N-dimensional arrays
import matplotlib.pyplot as plt
from pyrsistent import v  # data plotting
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import wandb
from cancerclassification.early_stopping import (
    EarlyStopping,
)

########################################################################################


class NetSwav(nn.Module):
    """Neural network for the pretraining with the SwAV algorithm.

    Args:
        input_dim (int): Inputs dimension.
        l1 (int, optional): Dimension of the first hidden layer. Defaults to 512.
        l2 (int, optional): Dimension of the second hidden layer. Defaults to 256.
        l3 (int, optional): Dimension of the third hidden layer. Defaults to 128.
        proj (int, optional): Dimension of the projection hidden layer. Defaults to 64.
        nproto (int, optional): Number of prototypes. Defaults to 60.
    """

    def __init__(self, input_dim, l1=512, l2=256, l3=128, proj=64, nproto=60):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, l1)
        self.fc1_bn = nn.BatchNorm1d(l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc2_bn = nn.BatchNorm1d(l2)
        self.fc3 = nn.Linear(l2, l3)
        self.fc3_bn = nn.BatchNorm1d(l3)
        self.projection = nn.Linear(l3, proj)
        self.prototypes = nn.Linear(proj, nproto, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = self.projection(x)
        x = nn.functional.normalize(x, dim=1, p=2)  # normalize the feature
        x = self.prototypes(x)
        return x


@torch.no_grad()
def mc_sinkhorn(out, eps):
    """Implementation of the Sinkhorn-Knopp algorithm

    Args:
        out (torch.Tensor): Prototypes assignment matrix
        eps (float): Labelling smothness parameter (lower is harder assignment).

    Returns:
        torch.Tensor: Codes matrix (optimized prototypes matrix)
    """
    epsilon = eps
    sinkhorn_iterations = 3
    Q = torch.exp(
        out / epsilon
    ).t()  # Q is K-by-B for consistency with notations from our paper
    K, B = Q.shape

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    # dist.all_reduce(sum_Q)
    Q /= sum_Q

    for _ in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        # dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


def fast_random_mask(X, prop, device):
    """Masks a defined proportion of the columns of a matrix with values in a gaussian.

    Args:
        X (torch.Tensor): Input matrix.
        prop (float): Proportion of the columns to mask (between 0 and 1).
        device (str): Identifier of the cuda device.

    Returns:
        torch.Tensor: Masked matrix.
    """
    n_dim = X.shape[1]  # size of the original array
    n = int(prop * n_dim)  # size of the random mask
    rng = np.random.default_rng()
    col_subset = rng.choice(n_dim, replace=False, size=n)
    new_X = torch.clone(X)
    new_X[:, col_subset] = torch.normal(0, 1, size=new_X[:, col_subset].shape).to(
        device
    )
    return new_X


def fast_noised_input(X, coeff, device):
    """Masks a matrix with a defined intensity of gaussian noise.

    Args:
        X (torch.Tensor): Input matrix.
        coeff (float): Coefficient multiplying the gaussian noise matrix before adding
            it to the input matrix.
        device (str): Identifier of the cuda device.

    Returns:
        torch.Tensor: Masked matrix.
    """
    new_X = torch.clone(X).to(device)
    new_X += coeff * torch.tile(torch.randn(new_X.shape[1]), (new_X.shape[0], 1)).to(
        device
    )
    return new_X


def vae_aug(X, coeff, vae):
    """Create a new (augmented) matrix by passing the input matrix in a specified VAE.

    Args:
        X (torch.Tensor): Input matrix.
        coeff (float): Coefficient multiplying the epsilon responsible for the shift of
            the z vector in the VAE's latent space. Should be greater than 1.
        vae (nn.Module): Neural network (model) of the trained VAE.

    Returns:
        torch.Tensor: Augmented matrix.
    """
    X, _, _ = vae(X, coeff)
    return X


def swav_epoch(net, dataloader, optimizer, device, coeff, eps, method, vae):
    """Trains the SwAV network (forward propagation, loss, back propagation) for one
        epoch.

    Args:
        net (nn.Module): Neural network (model) of the training.
        dataloader (Dataloader): Custom Pytorch Dataloader of the training dataset.
        optimizer (optim): Optimizer of the neural network.
        device (str): Identifier of the cuda device.
        coeff (float): Coefficient used for the data augmentation.
        eps (float): SwAV algorithm labelling smothness parameter (lower is harder
            assignment).
        method (str): Augmentation method. Value can be one of "noise", "mask" or "vae".
        vae (nn.Module): Neural network (model) of the trained VAE.

    Returns:
        (float, float): Cumulated sum of the losses over the mini-batches and number of
        mini-batches.
    """

    epoch_loss = 0
    train_steps = 0

    # Training
    net.train()
    for batch in dataloader:

        # get the inputs; batch is a list of [inputs, labels]
        X, _ = batch

        # perform the augmentation
        if method == "noise":
            X1 = fast_noised_input(X, coeff, device)
            X2 = fast_noised_input(X, coeff, device)
        if method == "mask":
            X1 = fast_random_mask(X, coeff, device)
            X2 = fast_random_mask(X, coeff, device)
        if method == "vae" and vae:
            X1 = vae_aug(X, coeff, vae)
            X2 = vae_aug(X, coeff, vae)
        if method == "vae_masked" and vae:
            X1 = vae_aug(X, coeff, vae)
            X1 = fast_random_mask(X1, coeff, device)
            X2 = vae_aug(X, coeff, vae)
            X2 = fast_random_mask(X2, coeff, device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # normalize the prototypes
        with torch.no_grad():
            w = net.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            net.prototypes.weight.copy_(w)

        # get the two feature vectors (z1 & z2)
        z1 = net(X1.view(-1, X.shape[1]))
        z2 = net(X2.view(-1, X.shape[1]))

        # compute the codes (q1 & q2)
        q1 = mc_sinkhorn(z1, eps)
        q2 = mc_sinkhorn(z2, eps)

        # compute the loss
        temp = 0.1
        loss = 0
        loss -= torch.mean(torch.sum(q1 * F.log_softmax(z2 / temp, dim=1), dim=1))
        loss -= torch.mean(torch.sum(q2 * F.log_softmax(z1 / temp, dim=1), dim=1))
        loss /= 2

        # finalize step
        epoch_loss += loss.item()
        train_steps += 1
        loss.backward()
        optimizer.step()

    return epoch_loss, train_steps


def swav_val_epoch(net, dataloader, device, coeff, eps, method, vae):
    """Evaluates the SwAV network (inference, loss) for one epoch.

    Args:
        net (nn.Module): Neural network (model) of the training.
        dataloader (Dataloader): Custom Pytorch Dataloader of the training dataset.
        device (str): Identifier of the cuda device.
        coeff (float): Coefficient used for the data augmentation.
        eps (float): SwAV algorithm labelling smothness parameter (lower is harder
            assignment).
        method (str): Augmentation method. Value can be one of "noise", "mask" or "vae".
        vae (nn.Module): Neural network (model) of the trained VAE.

    Returns:
        (float, float): Cumulated sum of the losses over the mini-batches and number of
        mini-batches.
    """

    epoch_loss = 0
    train_steps = 0

    # Training
    net.eval()
    for batch in dataloader:

        # get the inputs; batch is a list of [inputs, labels]
        X, _ = batch

        # perform the augmentation
        if method == "noise":
            X1 = fast_noised_input(X, coeff, device)
            X2 = fast_noised_input(X, coeff, device)
        if method == "mask":
            X1 = fast_random_mask(X, coeff, device)
            X2 = fast_random_mask(X, coeff, device)
        if method == "vae" and vae:
            X1 = vae_aug(X, coeff, vae)
            X2 = vae_aug(X, coeff, vae)

        with torch.no_grad():
            w = net.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            net.prototypes.weight.copy_(w)

        # get the two feature vectors (z1 & z2)
        z1 = net(X1.view(-1, X.shape[1]))
        z2 = net(X2.view(-1, X.shape[1]))

        # compute the codes (q1 & q2)
        q1 = mc_sinkhorn(z1, eps)
        q2 = mc_sinkhorn(z2, eps)

        # compute the loss
        temp = 0.1
        loss = 0
        loss -= torch.mean(torch.sum(q1 * F.log_softmax(z2 / temp, dim=1), dim=1))
        loss -= torch.mean(torch.sum(q2 * F.log_softmax(z1 / temp, dim=1), dim=1))
        loss /= 2

        # finalize step
        epoch_loss += loss.item()
        train_steps += 1

    return epoch_loss, train_steps


def train_swav_tcga(
    config,
    dataloaders,
    net,
    device,
    early_stop=0,
    val=True,
    log=True,
    method="mask",
    eps=0.05,
    vae=None,
):
    """Main function for the training and the evaluation of a neural network for the
        pretraining with the SwAV algorithm.

    Args:
        config (dict): Dictionnary containing the hyperparameters.
        dataloaders (tuple of Dataloader): Custom Pytorch Dataloaders of the train,
            validation and the test dataset.
        net (nn.Module): Neural network (model) of the training.
        device (str): Identifier of the cuda device.
        early_stop (int, optional): Patience of the early stopping of the training. To
            be set to 0 to disable early stopping. Defaults to 0.
        val (bool, optional): If True, performs an evaluation on the validation dataset
            at each epoch. Defaults to True.
        log (bool, optional): If True, the metrics related to the training are logged to
            the console. Defaults to True.
        method (str, optional): Augmentation method. Value can be one of "noise", "mask"
            or "vae".. Defaults to "mask".
        eps (float, optional): SwAV algorithm labelling smothness parameter (lower is
            harder assignment). Defaults to 0.05.
        vae (nn.Module, optional): Neural network (model) of the trained VAE. Defaults
            to None.

    Returns:
        _type_: _description_
    """

    swavset, valset, _ = dataloaders

    optimizer = optim.Adam(net.parameters(), lr=config["lr_init"])

    if early_stop:
        es = EarlyStopping(patience=early_stop)

    for epoch in range(config["epochs"]):
        train_loss, train_steps = swav_epoch(
            net,
            swavset,
            optimizer,
            device,
            coeff=config["coeff"],
            eps=eps,
            method=method,
            vae=vae,
        )
        if val:
            val_loss, val_steps = swav_val_epoch(
                net,
                valset,
                device,
                coeff=config["coeff"],
                eps=eps,
                method=method,
                vae=vae,
            )

        if early_stop and epoch > 5:
            es(val_loss / val_steps)
            if es.early_stop:
                break
        if log:
            print(
                "| Epoch: {:>3}/{} | SwAVTrainLoss={:.4f} |".format(
                    epoch + 1, config["epochs"], train_loss / train_steps
                ),
                end="",
            )
            if val:
                print(" SwAVEvalLoss={:.4f} |".format(val_loss / val_steps))
            else:
                print("")

    return train_loss / train_steps
