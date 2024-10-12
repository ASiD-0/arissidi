import joblib
import os
import random
import numpy as np
import pandas as pd
import sys
sys.path.append('./SFTC-Unlearn')
from utils.unlearning_alg.scrub import scrub
from utils.unlearning_alg.sftc_unlearn import sftc_unlearn
from utils.unlearning_alg.neg_grad import neg_grad
from utils.unlearning_utils import RandomDistributionGenerator, CustomPseudoLabelDataset
from utils.preprocessing import to_torch_loader
from utils.train_utils import predict_epoch
from utils.loss_utils import kl_loss, custom_kl_loss, SelectiveCrossEntropyLoss
import copy
import torchvision.models as models
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import datasets, transforms
from PIL import Image

'''
Default Functions
'''

def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Model for Images
class TargetModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(TargetModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer_size)  # First hidden layer
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size // 2)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_layer_size // 2, output_size)  # Output layer

    def forward(self, x):
        x = x.view(-1, input_size)  # Flatten input to (batch_size, input_size)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No activation for the output layer
        return x

# Model for Tabular
# class TargetModel(nn.Module):
#     def __init__(self, input_size, hidden_layer_size, output_size):
#         super(TargetModel, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_layer_size)
#         self.tanh_activation = nn.Tanh()
#         self.output_layer = nn.Linear(hidden_layer_size, output_size)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.tanh_activation(x)
#         x = self.output_layer(x)
#         return x


def create_membership_dataframe(
    model: nn.Module,
    member_data: pd.DataFrame,
    non_member_data: pd.DataFrame
) -> pd.DataFrame:
    """Create a DataFrame with model outputs and membership status. Also apply softmax on the outputs"""

    model.eval()
    member_tensor = torch.tensor(member_data.values, dtype=torch.float32)
    non_member_tensor = torch.tensor(non_member_data.values, dtype=torch.float32)

    member_outputs = F.softmax(model(member_tensor), dim=1)
    non_member_outputs = F.softmax(model(non_member_tensor), dim=1)

    member_df = pd.DataFrame(member_outputs.detach().numpy())
    member_df['membership'] = True

    non_member_df = pd.DataFrame(non_member_outputs.detach().numpy())
    non_member_df['membership'] = False

    membership_df = pd.concat([member_df, non_member_df], ignore_index=True)

    return membership_df


def create_merged_loader(X_retain, y_retain, X_forget, y_forget, batch_size=32, shuffle=True):
    """
    Create a merged DataLoader from retain and forget datasets where the inputs are PyTorch tensors.

    Args:
        X_retain (torch.Tensor): Features for the retain dataset (tensor).
        y_retain (torch.Tensor): Labels for the retain dataset (tensor).
        X_forget (torch.Tensor): Features for the forget dataset (tensor).
        y_forget (torch.Tensor): Labels for the forget dataset (tensor).
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data in the DataLoader.

    Returns:
        DataLoader: Merged DataLoader with features, labels, and pseudo-labels.
    """
    # Create pseudo-labels: 0 for retain, 1 for forget
    pseudo_labels_retain = torch.zeros(len(X_retain), dtype=torch.long)  # Pseudo-label 0 for retain
    pseudo_labels_forget = torch.ones(len(X_forget), dtype=torch.long)   # Pseudo-label 1 for forget

    # Combine retain and forget datasets
    X_merged = torch.cat([X_retain, X_forget], dim=0)
    y_merged = torch.cat([y_retain, y_forget], dim=0)
    pseudo_labels_merged = torch.cat([pseudo_labels_retain, pseudo_labels_forget], dim=0)

    # Create a TensorDataset with X, y, and pseudo-labels
    merged_dataset = TensorDataset(X_merged, y_merged, pseudo_labels_merged)

    # Create the DataLoader for the merged dataset
    merged_loader = DataLoader(merged_dataset, batch_size=batch_size, shuffle=shuffle)

    return merged_loader


def evaluate_attack_model_get_stats(dataset, model):
    """
    Evaluates and visualizes the performance of the attack model.

    Parameters:
        dataset (pd.DataFrame): DataFrame containing features and 'membership' column.
        attack_model (sklearn.base.ClassifierMixin): Trained attack model.
        plot_choice (str): Type of plot to display. 'roc' for ROC curve, 'confusion' for confusion matrix.

    Returns:
        dict: Dictionary containing precision, recall, accuracy, F1 score, and confusion matrix.
    """
    # Extract features and labels from the dataset
    features = dataset.drop(columns=['membership'])
    labels = dataset['membership']

    # Predict probabilities using the trained attack model
    predictions = model.predict(features)
    # Probabilities for the positive class
    probabilities = model.predict_proba(features)[:, 1]

    # Compute metrics
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    accuracy = model.score(features, labels)
    f1 = f1_score(labels, predictions)

    return accuracy, f1, precision, recall


def parsing(meta_data):
    image_age_list = []
    for idx, row in meta_data.iterrows():
        image_path = row['image_path']
        age_class = row['age_class']
        image_age_list.append([image_path, age_class])
    return image_age_list

# Custom Dataset class to handle image loading and transformations
class MUFACCustomDataset(Dataset):
    def __init__(self, meta_data, image_directory, transform=None, forget=False, retain=False):
        self.meta_data = meta_data
        self.image_directory = image_directory
        self.transform = transform

        # Process the metadata
        image_age_list = parsing(meta_data)

        self.image_age_list = image_age_list
        self.age_class_to_label = {
            "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7
        }

        # Splitting dataset for machine unlearning (forget and retain datasets)
        if forget:
            self.image_age_list = self.image_age_list[:1500]
        if retain:
            self.image_age_list = self.image_age_list[1500:]

    def __len__(self):
        return len(self.image_age_list)

    def __getitem__(self, idx):
        image_path, age_class = self.image_age_list[idx]
        img = Image.open(os.path.join(self.image_directory, image_path))
        label = self.age_class_to_label[age_class]

        if self.transform:
            img = self.transform(img)

        return img, label


'''
Unlearning Functions
'''


def scrub_tracking_MIA(
        retain_member_df: pd.DataFrame,
        forget_member_df: pd.DataFrame,
        non_member_df: pd.DataFrame,
        attack_model_trained: RandomForestClassifier,
        model: torch.nn.Module,
        retain_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        forget_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        criterion: torch.nn.Module,
        epochs: int = 3,
        return_history: bool = False,
        device: str = 'cuda',
        **kwargs
):
    """"""
    try:
        teacher_model = kwargs['teacher_model']
    except KeyError:
        raise ValueError("SCRUB requires the teacher_model to be given as input!")

    temperature = kwargs.get("temperature", 4.)

    teacher_model.eval()
    model.train()

    MIA_acc_retain, MIA_f1_retain, MIA_prec_retain, MIA_recall_retain = [], [], [], []
    MIA_acc_forget, MIA_f1_forget, MIA_prec_forget, MIA_recall_forget = [], [], [], []

    train_losses_retain, train_losses_forget = [], []
    val_losses, test_losses, forget_losses = [], [], []
    train_accs, val_accs, test_accs, forget_accs = [], [], [], []
    # Training with retain data
    for epoch in range(epochs):
        running_classification_loss_retain, running_kl_retain = [], []
        total_retain_loss, running_kl_forget = [], []

        model.train()
        for x_forget, y_forget in tqdm(forget_loader, desc=f"Epoch {epoch + 1} - Training on Forget"):
            x_forget, y_forget = x_forget.to(device), y_forget.to(device)

            # Make a prediction using the teacher
            with torch.no_grad():
                # local_x = x_forget.view(-1, 3, 32, 32)
                teacher_outputs = teacher_model(x_forget)


            optimizer.zero_grad()
            # Make a prediction using the student
            outputs = model(x_forget)
            # print('my model ', outputs.shape)

            # maximize the kl div loss
            loss = -kl_loss(model_logits=outputs,
                            teacher_logits=teacher_outputs,
                            temperature=temperature,
                            distill=True)
            loss.backward()
            optimizer.step()
            running_kl_forget.append(loss.item())

        model.train()

        for x, y in tqdm(retain_loader, desc=f"Epoch {epoch + 1} - Training"):
            x, y = x.to(device), y.to(device)

            # Make a prediction using the teacher
            with torch.no_grad():
                # temp_x = x.view(-1, 3, 32, 32)
                teacher_outputs = teacher_model(x)

            optimizer.zero_grad()
            # Make a prediction using the student
            outputs = model(x)

            loss = criterion(outputs, y)
            running_classification_loss_retain.append(loss.item())
            kl = kl_loss(model_logits=outputs,
                         teacher_logits=teacher_outputs,
                         temperature=temperature,
                         distill=True)
            running_kl_retain.append(kl.item())
            total_loss = loss + kl
            total_loss.backward()
            optimizer.step()
            total_retain_loss.append(total_loss.item())

        if scheduler is not None:
            scheduler.step()

        epoch_classification_loss = sum(running_classification_loss_retain) / len(running_classification_loss_retain)
        epoch_kl_retain_loss = sum(running_kl_retain) / len(running_kl_retain)
        epoch_kl_forget_loss = sum(running_kl_forget) / len(running_kl_forget)

        train_losses_retain.append(epoch_classification_loss)
        train_losses_forget.append(epoch_kl_forget_loss)

        _, train_acc = predict_epoch(model, retain_loader, criterion, device)
        val_loss, val_acc = predict_epoch(model, val_loader, criterion, device)
        test_loss, test_acc = predict_epoch(model, test_loader, criterion, device)

        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        if forget_loader is not None:
            forget_loss, forget_acc = predict_epoch(model, forget_loader, criterion, device)
            forget_losses.append(forget_loss)
            forget_accs.append(forget_acc)

        print(f"[Epoch {epoch + 1}]\n\t[Train]\tRetain Classification Loss={epoch_classification_loss:.4f}, "
              f"Retain KL Loss={epoch_kl_retain_loss:.4f}, Forget KL Loss={epoch_kl_forget_loss}, Acc={train_acc:.4f}\n\t"
              f"[Val]\tLoss={val_loss:.4f}, Acc={val_acc:.4f}\n\t"
              f"[Test]\tLoss={test_loss:.4f}, Acc={test_acc:.4f}")
        if forget_loader is not None:
            print(f"\t[Forget] Loss={forget_loss:.4f}, Acc={forget_acc:.4f}")

        dataset_from_model_outputs = create_membership_dataframe(model, retain_member_df, non_member_df)
        acc, f1, prec, recall = evaluate_attack_model_get_stats(dataset_from_model_outputs, attack_model_trained)
        MIA_acc_retain.append(acc)
        MIA_f1_retain.append(f1)
        MIA_prec_retain.append(prec)
        MIA_recall_retain.append(recall)

        dataset_from_model_outputs = create_membership_dataframe(model, forget_member_df, non_member_df)
        acc, f1, prec, recall = evaluate_attack_model_get_stats(dataset_from_model_outputs, attack_model_trained)
        MIA_acc_forget.append(acc)
        MIA_f1_forget.append(f1)
        MIA_prec_forget.append(prec)
        MIA_recall_forget.append(recall)

        losses = {"train": train_losses_retain,
                  "val": val_losses,
                  "test": test_losses}
        accs = {"train": train_accs,
                "val": val_accs,
                "test": test_accs}
        if forget_loader is not None:
            losses["forget"] = forget_losses
            accs["forget"] = forget_accs

        mia_stats = {"retain": {"acc": MIA_acc_retain,
                                "f1": MIA_f1_retain,
                                "precision": MIA_prec_retain,
                                "recall": MIA_recall_retain},
                     "forget": {"acc": MIA_acc_forget,
                                "f1": MIA_f1_forget,
                                "precision": MIA_prec_forget,
                                "recall": MIA_recall_forget}}            
    return mia_stats, accs, losses


def neg_grad_tracking_MIA(
            retain_member_df: pd.DataFrame,
            forget_member_df: pd.DataFrame,
            non_member_df: pd.DataFrame,
            attack_model_trained: RandomForestClassifier,
            model: torch.nn.Module,
            retain_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: DataLoader,
            forget_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler,
            criterion: torch.nn.Module,
            epochs: int = 5,
            return_history: bool = False,
            device: str = 'cuda',
            **kwargs
            ):
    advanced_neg_grad = kwargs.get("advanced_neg_grad", False)

    MIA_acc_retain, MIA_f1_retain, MIA_prec_retain, MIA_recall_retain = [], [], [], []
    MIA_acc_forget, MIA_f1_forget, MIA_prec_forget, MIA_recall_forget = [], [], [], []

    train_losses, val_losses, test_losses, forget_losses = [], [], [], []
    train_accs, val_accs, test_accs, forget_accs = [], [], [], []

    retain_iterator = iter(retain_loader)
    for epoch in range(epochs):
        model.train()

        running_loss = []
        for x_forget, y_forget in tqdm(forget_loader, desc=f"Epoch {epoch + 1} - Training"):
            x_forget, y_forget = x_forget.to(device), y_forget.to(device)

            try:
                x, y = next(retain_iterator)
            except StopIteration:
                retain_iterator = iter(retain_loader)
                x, y = next(retain_iterator)

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            # make a prediction on the forget set
            outputs_forget = model(x_forget)
            total_loss = -criterion(outputs_forget, y_forget)

            if advanced_neg_grad:
                # make a prediction on the retain set
                outputs = model(x)
                total_loss += criterion(outputs, y)

            total_loss.backward()
            optimizer.step()
            running_loss.append(total_loss.item())
        if scheduler is not None:
            scheduler.step()
        epoch_loss = sum(running_loss) / len(running_loss)
        train_losses.append(epoch_loss)
        _, train_acc = predict_epoch(model, retain_loader, criterion, device)
        val_loss, val_acc = predict_epoch(model, val_loader, criterion, device)
        test_loss, test_acc = predict_epoch(model, test_loader, criterion, device)

        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        if forget_loader is not None:
            forget_loss, forget_acc = predict_epoch(model, forget_loader, criterion, device)
            forget_losses.append(forget_loss)
            forget_accs.append(forget_acc)

        print(f"[Epoch {epoch + 1}]\n\t[Train]\tLoss={epoch_loss:.4f}, Acc={train_acc:.4f}\n\t"
              f"[Val]\tLoss={val_loss:.4f}, Acc={val_acc:.4f}\n\t"
              f"[Test]\tLoss={test_loss:.4f}, Acc={test_acc:.4f}")
        if forget_loader is not None:
            print(f"\t[Forget] Loss={forget_loss:.4f}, Acc={forget_acc:.4f}")


        dataset_from_model_outputs = create_membership_dataframe(model, retain_member_df, non_member_df)
        acc, f1, prec, recall = evaluate_attack_model_get_stats(dataset_from_model_outputs, attack_model_trained)
        MIA_acc_retain.append(acc)
        MIA_f1_retain.append(f1)
        MIA_prec_retain.append(prec)
        MIA_recall_retain.append(recall)

        dataset_from_model_outputs = create_membership_dataframe(model, forget_member_df, non_member_df)
        acc, f1, prec, recall = evaluate_attack_model_get_stats(dataset_from_model_outputs, attack_model_trained)
        MIA_acc_forget.append(acc)
        MIA_f1_forget.append(f1)
        MIA_prec_forget.append(prec)
        MIA_recall_forget.append(recall)

        losses = {"train": train_losses,
                  "val": val_losses,
                  "test": test_losses}
        accs = {"train": train_accs,
                "val": val_accs,
                "test": test_accs}
        if forget_loader is not None:
            losses["forget"] = forget_losses
            accs["forget"] = forget_accs

        mia_stats = {"retain": {"acc": MIA_acc_retain,
                                "f1": MIA_f1_retain,
                                "precision": MIA_prec_retain,
                                "recall": MIA_recall_retain},
                     "forget": {"acc": MIA_acc_forget,
                                "f1": MIA_f1_forget,
                                "precision": MIA_prec_forget,
                                "recall": MIA_recall_forget}}            
    return mia_stats, accs, losses


def sftc_unlearn_tracking_MIA(
        retain_member_df: pd.DataFrame,
        forget_member_df: pd.DataFrame,
        non_member_df: pd.DataFrame,
        attack_model_trained: RandomForestClassifier,
        model: torch.nn.Module,
        retain_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        forget_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        criterion: torch.nn.Module,
        epochs: int = 3,
        return_history: bool = False,
        device: str = 'cuda',
        **kwargs
):
    try:
        merged_loader = kwargs['merged_loader']
    except KeyError:
        raise ValueError("SFTC requires the merged loader to be given as input!")
    try:
        teacher_model = kwargs['teacher_model']
    except KeyError:
        raise ValueError("SFTC requires the teacher_model to be given as input!")
    try:
        dummy_model = kwargs['dummy_model']
    except KeyError:
        raise ValueError("SFTC requires the dummy_model to be given as input!")

    temperature = kwargs.get("temperature", 4.)
    confuse_fraction = kwargs.get("confuse_fraction", 0.)

    labels = set()
    for _, label in val_loader:
        labels.update(label.numpy())
    num_classes = len(labels)

    selective_criterion = SelectiveCrossEntropyLoss()

    teacher_model.eval()
    if not isinstance(dummy_model, RandomDistributionGenerator):
        dummy_model.eval()

    model.train()

    MIA_acc_retain, MIA_f1_retain, MIA_prec_retain, MIA_recall_retain = [], [], [], []
    MIA_acc_forget, MIA_f1_forget, MIA_prec_forget, MIA_recall_forget = [], [], [], []

    train_losses_retain, train_losses_forget = [], []
    val_losses, test_losses, forget_losses = [], [], []
    train_accs, val_accs, test_accs, forget_accs = [], [], [], []

    # Training with retain data
    for epoch in range(epochs):
        model.train()
        running_cross_entropy_retain = []
        running_kl = []
        for x, y, pseudo_label in tqdm(merged_loader, desc=f"Epoch {epoch + 1} - Training"):
            x, y, pseudo_label = x.to(device), y.to(device), pseudo_label.to(device)

            # Make a prediction using the teacher
            with torch.no_grad():
                teacher_outputs = teacher_model(x)

                # Confusion
                if confuse_fraction == 0.:
                    if isinstance(dummy_model, RandomDistributionGenerator):
                        dummy_outputs = dummy_model(len(x), y)
                    else:
                        raise ValueError("Cannot perform confusion with fraction=0. and a trained model!"
                                         " Please use confuse_fraction=1.")
                elif confuse_fraction == 1.:
                    if isinstance(dummy_model, RandomDistributionGenerator):
                        dummy_outputs = dummy_model(len(x), None)
                        dummy_outputs = dummy_outputs.to(device)
                    else:
                        dummy_outputs = dummy_model(x)
                else:
                    # Select indices where pseudo_label equals to 1
                    forget_indices = torch.where(pseudo_label == 1)[0]
                    # Make a random permutation
                    permuted_indices = forget_indices[torch.randperm(len(forget_indices))]
                    n = len(forget_indices)
                    replace_n = int(n * confuse_fraction)
                    if replace_n == 0:
                        replace_n = 1
                    fake_indices = torch.randperm(n)[:replace_n]
                    indices = permuted_indices[fake_indices]
                    tmp_y = copy.deepcopy(y)
                    tmp_y[indices] = torch.randint(0, num_classes, (replace_n,), device=y.device)

                    if isinstance(dummy_model, RandomDistributionGenerator):
                        dummy_outputs = dummy_model(len(x), tmp_y)
                    else:
                        raise ValueError(f"Cannot perform confusion with fraction={confuse_fraction} "
                                         f"and a trained model! Please use confuse_fraction=1.")

            optimizer.zero_grad()
            # Make a prediction using the student
            outputs = model(x)

            selective_cross_entropy = selective_criterion(outputs, y, pseudo_label, 0)

            kl_div = custom_kl_loss(
                teacher_logits=teacher_outputs,
                dummy_logits=dummy_outputs,
                student_logits=outputs,
                pseudo_labels=pseudo_label,
                kl_temperature=temperature
            )
            running_cross_entropy_retain.append(selective_cross_entropy.item())
            running_kl.append(kl_div.item())

            total_loss = selective_cross_entropy + kl_div
            total_loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        epoch_cross_entropy = sum(running_cross_entropy_retain) / len(running_cross_entropy_retain)
        epoch_kl_retain_loss = sum(running_kl) / len(running_kl)

        _, train_acc = predict_epoch(model, retain_loader, criterion, device)
        val_loss, val_acc = predict_epoch(model, val_loader, criterion, device)
        test_loss, test_acc = predict_epoch(model, test_loader, criterion, device)

        train_losses_retain.append(epoch_cross_entropy)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        if forget_loader is not None:
            forget_loss, forget_acc = predict_epoch(model, forget_loader, criterion, device)
            forget_losses.append(forget_loss)
            forget_accs.append(forget_acc)

        print(f"[Epoch {epoch + 1}]\n\t[Train]\t"
              f"Retain Loss={epoch_cross_entropy:.4f}, Retain Acc={train_acc:.4f}, KL Loss={epoch_kl_retain_loss:.4f}\n"
              f"\t[Val]\tLoss={val_loss:.4f}, Acc={val_acc:.4f}\n\t"
              f"[Test]\tLoss={test_loss:.4f}, Acc={test_acc:.4f}")
        if forget_loader is not None:
            print(f"\t[Forget] Loss={forget_loss:.4f}, Acc={forget_acc:.4f}")


        dataset_from_model_outputs = create_membership_dataframe(model, retain_member_df, non_member_df)
        acc, f1, prec, recall = evaluate_attack_model_get_stats(dataset_from_model_outputs, attack_model_trained)
        MIA_acc_retain.append(acc)
        MIA_f1_retain.append(f1)
        MIA_prec_retain.append(prec)
        MIA_recall_retain.append(recall)

        dataset_from_model_outputs = create_membership_dataframe(model, forget_member_df, non_member_df)
        acc, f1, prec, recall = evaluate_attack_model_get_stats(dataset_from_model_outputs, attack_model_trained)
        MIA_acc_forget.append(acc)
        MIA_f1_forget.append(f1)
        MIA_prec_forget.append(prec)
        MIA_recall_forget.append(recall)

        losses = {"train": train_losses_retain,
                  "val": val_losses,
                  "test": test_losses}
        accs = {"train": train_accs,
                "val": val_accs,
                "test": test_accs}
        if forget_loader is not None:
            losses["forget"] = forget_losses
            accs["forget"] = forget_accs

        mia_stats = {"retain": {"acc": MIA_acc_retain,
                                "f1": MIA_f1_retain,
                                "precision": MIA_prec_retain,
                                "recall": MIA_recall_retain},
                     "forget": {"acc": MIA_acc_forget,
                                "f1": MIA_f1_forget,
                                "precision": MIA_prec_forget,
                                "recall": MIA_recall_forget}}            
    return mia_stats, accs, losses

'''
Datasets
'''

def get_texas_100_dataset(path='data/texas100.npz', limit_rows=None):
    """
    Processes the Texas 100 dataset.

    Returns:
    X (DataFrame): DataFrame of features.
    y (DataFrame): DataFrame with a single 'label' column representing the class index.
    num_features (int): Number of features.
    num_classes (int): Number of unique classes in the dataset.
    """

    data = np.load(path)
    features = data['features']
    labels = data['labels']

    if limit_rows:
        features = features[:limit_rows]
        labels = labels[:limit_rows]

    # Convert features and labels to DataFrames if they are not already
    X = pd.DataFrame(features)

    # Convert one-hot encoded labels to class indices
    y = pd.DataFrame()
    y['label'] = pd.DataFrame(labels).idxmax(axis=1).astype(int)

    # Get the number of features and classes
    num_features = X.shape[1]
    num_classes = len(np.unique(y['label']))

    print(f"Number of features: {num_features}")
    print(f"Number of classes: {num_classes}")

    return X, y, num_features, num_classes


def get_MUFAC_dataset(train_meta_data_path, train_image_directory, batch_size=64, percentage_of_rows_to_drop=0.2):
    # Define transformations for training data
    train_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()
    ])

    # Load metadata, limiting to 2000 rows
    train_meta_data = pd.read_csv(train_meta_data_path)

    # Initialize the Dataset and DataLoader
    train_dataset = MUFACCustomDataset(train_meta_data, train_image_directory, train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Convert DataLoader to Pandas DataFrames (dataloader_to_dataframe logic)
    all_images = []
    all_labels = []

    for images, labels in train_dataloader:
        # Flatten the image tensors (e.g., 64, 3, 128, 128 -> 64, 3*128*128)
        flattened_images = images.view(images.size(0), -1).numpy()

        # Convert labels to numpy array
        labels = labels.numpy()

        # Accumulate images and labels
        all_images.append(flattened_images)
        all_labels.append(labels)

    # Concatenate all batches into a single array
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Create DataFrames
    X_train = pd.DataFrame(all_images)
    y_train = pd.DataFrame(all_labels, columns=['label'])

    # limit dataset size
    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=percentage_of_rows_to_drop, random_state=42, stratify=y_train)

    # Calculate the number of features and classes
    num_features = X_train.shape[1]
    num_classes = y_train['label'].nunique()

    return X_train, y_train, num_features, num_classes


def get_cifar10_dataset():
    datasets.CIFAR10.url="https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"

    # Define transformations (convert to tensor and normalize the images)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images for RGB channels
    ])


    # Load the CIFAR-10 dataset
    cifar10_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Convert to features (X) and targets (y)
    X = np.array([np.transpose(img.numpy(), (1, 2, 0)).flatten() for img, _ in cifar10_dataset])  # Flatten RGB images
    y = np.array([label for _, label in cifar10_dataset])

    # Convert to Pandas DataFrame
    X_df = pd.DataFrame(X, columns=[f'pixel_{i}' for i in range(X.shape[1])])
    y_df = pd.DataFrame(y, columns=['label'])

    num_features = X_df.shape[1]  # Number of features (pixels)
    num_classes = y_df['label'].nunique()  # Number of unique classes (0-9)

    return X_df, y_df, num_features, num_classes


def get_purchase_dataset(dataset_path='/content/dataset_purchase.csv', keep_rows=50_000):
    # Load the dataset, restricting to first 50,000 rows
    purchase_data = pd.read_csv(dataset_path).head(keep_rows)

    # Extract features (X) and adjust target labels (y)
    X = purchase_data.drop(columns=purchase_data.columns[0], axis=1)
    y = purchase_data.iloc[:, 0] - 1  # Adjust class labels from 1-100 to 0-99

    num_features = X.shape[1]  # Number of features (columns in X)
    num_classes = y.nunique()  # Number of unique classes in y

    return X, y, num_features, num_classes


if __name__ == "__main__":
    # load pre-trained models

    # target_model = torch.load('models/cifar_target_model.pth' )
    # attack_model = joblib.load("models/cifar_attack_model.jolib")

    # target_model = torch.load('models/purchase_target_model.pth' )
    # attack_model = joblib.load("models/purchase_attack_model.jolib")

    # target_model = torch.load('models/texas_100_target_model.pth' )
    # attack_model = joblib.load('models/texas_100_attack_model.jolib')

    target_model = torch.load('models/mufac_target_model.pth' )
    attack_model = joblib.load("models/mufac_attack_model.jolib")
    
    """
    Data preprocessing
    """
    # load the dataset
    set_random_seed(42)

    # X, y, num_features, num_classes = get_cifar10_dataset()
    # X, y, num_features, num_classes = get_purchase_dataset(dataset_path='data/dataset_purchase.csv', keep_rows=40_000)
    # X, y, num_features, num_classes = get_MUFAC_dataset("./data/mufac-128/custom_train_dataset.csv", "./data/mufac-128/train_images", percentage_of_rows_to_drop = 0.4)
    X, y, num_features, num_classes = get_texas_100_dataset(path='data/texas100.npz', limit_rows=40_000)
    print(X.shape)
    exit()

    input_size = num_features 
    output_size = num_classes

    """
    Split data into Target training data, Shadow training data and data as Non Member to test the Attack model
    """

    # Split the data into 80% temp and 20% test
    X_temp, X_test_MIA, y_temp, y_test_MIA = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Now, split the remaining 80% data into 40% target train and 40% shadow train
    X_target, X_shadow, y_target, y_shadow = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    print(X.shape)
    print(X_target.shape)

    X_train, X_test, y_train, y_test = train_test_split(X_target, y_target, test_size=0.2, random_state=42)

    # Keep this to create a test dataset for the attack model
    X_target_train_set = X_train
    y_target_train_set = y_train

    """ 
    Run MIA during each unlearning epoch
    """

    unlearned_model = copy.deepcopy(target_model)
    teacher_model = copy.deepcopy(target_model)
    dummy_model = RandomDistributionGenerator(dist='normal', dimensions=num_classes)

    # Define the forget set to forget class 7
    X_retain, X_forget, y_retain, y_forget = train_test_split(X_target_train_set, y_target_train_set, test_size=0.2, random_state=42, stratify=y_target_train_set)

    # on CIFAR10 data specifically unlearn only class 7 
    # y_forget = y_forget[y_forget['label'] == 7]
    # X_forget = X_forget.loc[y_forget.index]


    X_forget_tensor = torch.tensor(X_forget.values, dtype=torch.float32)
    y_forget_tensor = torch.tensor(y_forget.values, dtype=torch.long).squeeze()

    X_retain_tensor = torch.tensor(X_retain.values, dtype=torch.float32)
    y_retain_tensor = torch.tensor(y_retain.values, dtype=torch.long).squeeze()

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).squeeze()

    # Merge Loader only for sftc
    merged_loader = create_merged_loader(X_retain_tensor, y_retain_tensor, X_forget_tensor, y_forget_tensor, batch_size=32, shuffle=True)

    # Hyperparameters for Unlearning
    learning_rate = 0.0003
    epochs = 200
    batch_size = 32

    # Create DataLoader for forget set and test set
    forget_data = TensorDataset(X_forget_tensor, y_forget_tensor)
    forget_loader = DataLoader(forget_data, batch_size=batch_size, shuffle=True)

    retain_data = TensorDataset(X_retain_tensor, y_retain_tensor)
    retain_loader = DataLoader(retain_data, batch_size=batch_size, shuffle=True)

    test_data = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Initialize the model, loss function, and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(unlearned_model.parameters(), lr=learning_rate, momentum=0.9)

    sample_size = min(len(X_test_MIA), len(X_forget), len(X_retain))

    # sample_size = min(len(X_test_MIA), len(X_retain))
    retain_member_df = X_retain.sample(n=sample_size, replace=False)

    # sample_size = min(len(X_test_MIA), len(X_forget))
    forget_member_df = X_forget.sample(n=sample_size, replace=False)

    train_non_member_data = X_test_MIA.sample(n=sample_size, replace=False)

    # Unlearn the target neural network
    mia_stats, accs, losses = scrub_tracking_MIA(retain_member_df, forget_member_df, train_non_member_data, attack_model,  unlearned_model, retain_loader, test_loader, test_loader, forget_loader, optimizer=optimizer, scheduler=None, criterion=criterion, epochs=epochs, device=None, teacher_model=teacher_model)
    # mia_stats, accs, losses = neg_grad_tracking_MIA(retain_member_df, forget_member_df, train_non_member_data, attack_model, unlearned_model, retain_loader, test_loader, test_loader, forget_loader, optimizer=optimizer, scheduler=None, criterion=criterion, epochs=epochs, device=None)
    # mia_stats, accs, losses = sftc_unlearn_tracking_MIA(retain_member_df, forget_member_df, train_non_member_data, attack_model, unlearned_model, retain_loader, test_loader, test_loader, forget_loader, optimizer=optimizer, scheduler=None, criterion=criterion, epochs=epochs, device=None, merged_loader=merged_loader, teacher_model=teacher_model, dummy_model=dummy_model)

    '''
    Plots
    '''

    # Prepare the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each accuracy curve
    ax.plot(accs['train'], label='Train Accuracies', linestyle='-.', color='blue')
    ax.plot(accs['val'], label='Val Accuracies', linestyle='--', color='green')
    ax.plot(accs['test'], label='Test Accuracies', linestyle='-.', color='orange')
    ax.plot(accs['forget'], label='Forget Accuracies', linestyle='--', color='red')

    # Add grid and minor ticks
    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth=0.25)

    # Set labels, title, and legend
    ax.set_xlabel('Number of Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Curves')

    ax.legend(loc='lower left')

    # Save the plot
    plt.savefig('curves/accuracy_curves.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(mia_stats['forget']['acc'], label='Forget Accuracies', linestyle='-')
    ax.plot(mia_stats['retain']['acc'], label='Retain Accuracies', linestyle='--')
    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth=0.25)
    ax.set_xlabel('Number of Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('MIA Accuracy as the Target Model Unlearns')

    ax.legend(loc='lower left')

    # Save the plot
    plt.savefig('curves/mia_accuracy.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(mia_stats['forget']['f1'], label='Forget F1 Scores', linestyle='-')
    ax.plot(mia_stats['retain']['f1'], label='Retain F1 Scores', linestyle='--')
    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth=0.25)
    ax.set_xlabel('Number of Epochs')
    ax.set_ylabel('F1 Score')
    ax.set_title('MIA F1 Score as the Target Model Unlearns')
    ax.legend(loc='lower left')

    # Save the plot
    plt.savefig('curves/mia_f1_score.pdf')
    plt.close(fig)


    # Plot for Precision
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(mia_stats['forget']['precision'], label='Forget Precision Scores', linestyle='-')
    ax.plot(mia_stats['retain']['precision'], label='Retain Precision Scores', linestyle='--')
    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth=0.25)
    ax.set_xlabel('Number of Epochs')
    ax.set_ylabel('Precision')
    ax.set_title('MIA Precision as the Target Model Unlearns')
    ax.legend(loc='lower left')

    # Save the Precision plot
    plt.savefig('curves/mia_precision.pdf')
    plt.close(fig)

    # Plot for Recall
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(mia_stats['forget']['recall'], label='Forget Recall Scores', linestyle='-')
    ax.plot(mia_stats['retain']['recall'], label='Retain Recall Scores', linestyle='--')
    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth=0.25)
    ax.set_xlabel('Number of Epochs')
    ax.set_ylabel('Recall')
    ax.set_title('MIA Recall as the Target Model Unlearns')
    ax.legend(loc='lower left')

    # Save the Recall plot
    plt.savefig('curves/mia_recall.pdf')
    plt.close(fig)

    epochs = list(range(0, len(mia_stats['forget']['acc'])))
    forget_acc = mia_stats['forget']['acc']
    retain_acc = mia_stats['retain']['acc']
    forget_f1 = mia_stats['forget']['f1']
    retain_f1 = mia_stats['retain']['f1']
    forget_precision = mia_stats['forget']['precision']
    retain_precision = mia_stats['retain']['precision']
    forget_recall = mia_stats['forget']['recall']
    retain_recall = mia_stats['retain']['recall']

    # Create a DataFrame
    df = pd.DataFrame({
        'Epoch': epochs,
        'MIA Forget Acc': forget_acc,
        'MIA Retain Acc': retain_acc,
        'MIA Forget F1': forget_f1,
        'MIA Retain F1': retain_f1,
        'MIA Forget Precision': forget_precision,
        'MIA Retain Precision': retain_precision,
        'MIA Forget Recall': forget_recall,
        'MIA Retain Recall': retain_recall,
        'Train Acc': accs['train'],
        'Val Acc': accs['val'],
        'Test Acc': accs['test'],
        'Forget Acc': accs['forget'],
    })

    # Define the CSV file name
    csv_file = 'curves/unlearn_algo_mia_stats.csv'

    # Write to CSV
    df.to_csv(csv_file, index=False)
