from cancerclassification.early_stopping import (
    EarlyStopping,
)  # manipulate N-dimensional arrays
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
from lifelines.utils import concordance_index
import torch

########################################################################################


class CancerDatasetSurvival(Dataset):
    """Custom Pytorch dataset to work with the TCGA data on the specific survival task.

    Args:
        survival (numpy.ndarray): Time to event (the event being the last check-up if
            the patient is alive or, if not, the patient's death).
        vital (numpy.ndarray): Vital status of the patient (1: dead, 0: alive)
        inputs (numpy.ndarray): 2-dimensional array of the inputs of the neural network.
        device (_type_, optional): Identifier of the cuda device. Defaults to None.
    """

    def __init__(self, survival, vital, inputs, device=None):

        self.survival = torch.from_numpy(survival).to(dtype=torch.float32)
        self.status = torch.from_numpy(vital).to(dtype=torch.float32)
        self.inputs = torch.from_numpy(inputs).to(dtype=torch.float32)
        if device:
            self.survival = self.survival.to(device)
            self.status = self.status.to(device)
            self.inputs = self.inputs.to(device)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        survival = self.survival[idx]
        status = self.status[idx]
        inputs = self.inputs[idx]
        return survival, status, inputs


class NetSurvival(nn.Module):
    """Basic neural network for survival analysis task.

    Args:
        input_dim (int): Inputs dimension.
        l1 (int, optional): Dimension of the first hidden layer. Defaults to 512.
        l2 (int, optional): Dimension of the second hidden layer. Defaults to 256.
        l3 (int, optional): Dimension of the third hidden layer. Defaults to 128.
        do (float, optional): Dropout rate. Defaults to 0.25.
    """

    def __init__(self, input_dim, l1=512, l2=256, l3=128, do=0.25):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, l1)
        self.fc1_bn = nn.BatchNorm1d(l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc2_bn = nn.BatchNorm1d(l2)
        self.fc3 = nn.Linear(l2, l3)
        self.fc3_bn = nn.BatchNorm1d(l3)
        self.fc4s = nn.Linear(l3, 1)
        self.dropout = nn.Dropout(do)

    def forward(self, x):
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4s(x)
        return x


def neg_partial_likelihood(survival, vital, risk):
    """Computes the negative partial likelihood

    Args:
        survival (torch.tensor): Time to event (the event being the last check-up if
            the patient is alive or, if not, the patient's death).
        vital (torch.tensor): Vital status of the patient (1: dead, 0: alive).
        risk (torch.tensor): Risk scores.

    Returns:
        torch.tensor: Negative partial likelihoods.
    """

    idx = torch.argsort(survival, descending=True)
    risk = risk[idx]
    vital = vital[idx]
    log_risk = torch.logcumsumexp(risk, dim=0)
    uncensored_likelihood = risk - log_risk
    censored_likelihood = uncensored_likelihood * vital
    num_observed_events = torch.sum(vital)
    return -torch.sum(censored_likelihood) / num_observed_events


def train_epoch_survival(net, dataloader, optimizer, loss_function):
    """Trains the network (forward propagation, loss, back propagation) for one epoch.

    Args:
        net (nn.Module): Neural network (model) of the training.
        dataloader (Dataloader): Custom Pytorch Dataloader of the training dataset.
        optimizer (optim): Optimizer of the neural network.
        loss_function (function): Loss function of the neural network.

    Returns:
        (float, float): Cumulated sum of the losses over the mini-batches and number of
        mini-batches.
    """

    epoch_loss = 0
    train_steps = 0

    net.train()
    for batch in dataloader:
        survival, status, X = batch
        optimizer.zero_grad()
        risk = net(X.view(-1, X.shape[1]))
        loss = loss_function(survival, status, risk)
        epoch_loss += loss.item()
        train_steps += 1
        loss.backward()
        optimizer.step()

    return epoch_loss, train_steps


def eval_epoch_survival(net, dataloader, loss_function):
    """Evaluates the network (inference, loss, performance) on the validation or the
    test set.

    Args:
        net (nn.Module): Neural network (model) of the training.
        dataloader (Dataloader): Custom Pytorch Dataloader of the validation or the test
            dataset.
        loss_function (function): Loss function of the neural network.

    Returns:
        (float, float): Loss of the batch and concordance index value.
    """

    net.eval()
    with torch.no_grad():
        val_data = next(iter(dataloader))
        survival, status, X = val_data
        risk = net(X.view(-1, X.shape[1]))
        eval_loss = loss_function(survival, status, risk).cpu().detach().numpy()
        partial_hazard = torch.exp(risk)
        CI = concordance_index(
            survival.cpu().detach().numpy(),
            torch.neg(partial_hazard).cpu().detach().numpy(),
            status.cpu().detach().numpy(),
        )

    return eval_loss, CI


def train_nn_tcga_survival(
    config,
    dataloaders,
    net,
    early_stop=0,
    val=True,
    test=True,
    log=True,
    logger=None,
):
    """Main function for the training and the evaluation of a neural network for the
    survival analysis task.

    Args:
        config (dict): Dictionnary containing the hyperparameters.
        dataloaders (tuple of Dataloader): Custom Pytorch Dataloaders of the train,
            validation and the test dataset.
        net (nn.Module): Neural network (model) of the training.
        early_stop (int, optional): Patience of the early stopping of the training. To
            be set to 0 to disable early stopping. Defaults to 0.
        val (bool, optional): If True, performs an evaluation on the validation dataset
            at each epoch. Defaults to True.
        test (bool, optional): If True, performs an evaluation on the test dataset at
            each epoch. Defaults to True. Defaults to True.
        log (bool, optional): If True, the metrics related to the training are logged to
            the console. Defaults to True.
        logger (LogResults, optional): Instance of the class LogResults. If not
            specified, logging is disabled. Defaults to None.
    """

    trainset, valset, testset = dataloaders

    loss_function = (
        neg_partial_likelihood  # weighted loss to deal with th imbalance of the dataset
    )

    optimizer = optim.Adam(
        net.parameters(), lr=config["lr_init"], weight_decay=config["weight_decay"]
    )

    if early_stop:
        es = EarlyStopping(patience=early_stop, min_delta=0.001, log=log)

    for epoch in range(config["epochs"]):
        train_loss, train_steps = train_epoch_survival(
            net, trainset, optimizer, loss_function
        )
        if val:
            val_loss, val_CI = eval_epoch_survival(net, valset, loss_function)
        if test:
            test_loss, test_CI = eval_epoch_survival(net, testset, loss_function)
        if logger:
            logger.log_epoch(
                epoch=epoch,
                valacc=val_CI,
                valloss=val_loss,
                testacc=test_CI,
                testloss=test_loss,
            )
        if log:
            print(
                "| Epoch: {:>3}/{} | TrainLoss={:.4f} |".format(
                    epoch + 1,
                    config["epochs"],
                    train_loss / train_steps,
                ),
                end="",
            )
            if val:
                print(
                    " ValLoss={:.4f} | ValCI={:.4f} |".format(
                        val_loss,
                        val_CI,
                    ),
                    end="",
                )
            if test:
                print(
                    " TestLoss={:.4f} | TestCI={:.4f} |".format(
                        test_loss,
                        test_CI,
                    )
                )
            else:
                print("")

        if early_stop and epoch > 5:
            es(-val_CI)
            if es.early_stop:
                break
