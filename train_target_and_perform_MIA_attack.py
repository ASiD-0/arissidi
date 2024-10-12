import joblib
import os
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import datasets, transforms
from PIL import Image

def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Models for image data
# class TargetModel(nn.Module):
#     def __init__(self, input_size, hidden_layer_size, output_size):
#         super(TargetModel, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_layer_size)  # First hidden layer
#         self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size // 2)  # Second hidden layer
#         self.fc3 = nn.Linear(hidden_layer_size // 2, output_size)  # Output layer

#     def forward(self, x):
#         x = x.view(-1, input_size)  # Flatten input to (batch_size, input_size)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)  # No activation for the output layer
#         return x


# class  ShadowModel(nn.Module):
#     def __init__(self, input_size, hidden_layer_size, output_size):
#         super( ShadowModel, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_layer_size)  # First hidden layer
#         self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size // 2)  # Second hidden layer
#         self.fc3 = nn.Linear(hidden_layer_size // 2, output_size)  # Output layer

#     def forward(self, x):
#         x = x.view(-1, input_size)  # Flatten input to (batch_size, input_size)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)  # No activation for the output layer
#         return x

# Models for tabular data
class TargetModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(TargetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x

class ShadowModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(ShadowModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    patience: int,
    early_stopping: bool = True
) -> None:
    best_validation_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        model.train()
        total_training_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_training_loss += loss.item()

        average_training_loss = total_training_loss / len(train_loader)

        if early_stopping:
            model.eval()
            total_validation_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    total_validation_loss += loss.item()
            average_validation_loss = total_validation_loss / \
                len(val_loader)

            if average_validation_loss < best_validation_loss:
                best_validation_loss = average_validation_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("\nEarly stopping triggered.")
                    break
            print(f'\rEpoch {epoch + 1}/{num_epochs} | Train Loss: {average_training_loss:.4f}, '
                  f'Validation Loss: {average_validation_loss:.4f}', end='', flush=True)
        else:
            print(f'\rEpoch {epoch + 1}/{num_epochs} | Train Loss: {average_training_loss:.4f}',
                  end='', flush=True)


def evaluate_model_performance(model, loss_function, X_tensor, y_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        loss = loss_function(outputs, y_tensor)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        true_labels = y_tensor.cpu().numpy()
        accuracy = (predictions == true_labels).mean()
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    return loss.item(), accuracy, precision, recall


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


def train_attack_model_on_output_data(
    shadow_model_outputs_df: pd.DataFrame,
    random_seed: int = 42,
    test_model_capabilities: bool = False,
    verbose: bool = True,
) -> RandomForestClassifier:
    """
    Train an attack model on the output data of a shadow model.

    Args:
        shadow_model_outputs_df (pd.DataFrame): DataFrame containing the shadow model outputs.
        random_seed (int): Random seed for reproducibility.
        test_model_capabilities (bool): Flag to indicate if the model should be evaluated.

    Returns:
        RandomForestClassifier: The trained attack model.
    """
    # Shuffle the DataFrame and reset index
    training_df = shuffle(shadow_model_outputs_df, random_state=random_seed).reset_index(drop=True)

    # Prepare features and labels
    features = training_df.drop(columns=['membership'])
    labels = training_df['membership']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=random_seed)

    verbose_val = 2 if verbose else 0

    # Initialize and train the attack model
    attack_model = RandomForestClassifier(n_estimators=100, verbose=verbose_val, n_jobs=-1, random_state=random_seed)
    attack_model.fit(X_train, y_train)

    if test_model_capabilities:
        # Predict and evaluate the model
        y_pred = attack_model.predict(X_test)
        test_precision = precision_score(y_test, y_pred)
        test_accuracy = attack_model.score(X_test, y_test)
        print('Attack Model Evaluation:')
        print(f"Test Precision: {test_precision}")
        print(f"Test Accuracy: {test_accuracy}")

    return attack_model


def evaluate_attack_model(dataset, model, plot_title='Chart', plot_choice='roc'):
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

    # Compute confusion matrix
    conf_matrix = confusion_matrix(labels, predictions)
    conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True) * 100

    # ROC Curve calculation
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)

    # Plot based on the chosen type
    if plot_choice == 'roc':
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(plot_title)
        plt.legend(loc='lower right')
        plt.show()

    elif plot_choice == 'confusion':
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix_percent, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix (Percentages)')
        plt.show()

    # Print metrics
    print(f"Confusion Matrix (Percentages):\n{conf_matrix_percent}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")


def parsing(meta_data):
    image_age_list = []
    for idx, row in meta_data.iterrows():
        image_path = row['image_path']
        age_class = row['age_class']
        image_age_list.append([image_path, age_class])
    return image_age_list


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
DATASET FUNCTIONS
'''
def get_mnist_dataset():
    # Define transformations (convert to tensor and normalize the images)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize images with mean=0.5 and std=0.5
    ])

    # Load the dataset
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Convert to features (X) and targets (y)
    X = mnist_dataset.data.numpy()  # Convert to NumPy array
    y = mnist_dataset.targets.numpy()  # Convert to NumPy array

    # Reshape X to be 2D (each row will be a flattened 28x28 image)
    X = X.reshape(X.shape[0], -1)

    # Convert to Pandas DataFrame
    X_df = pd.DataFrame(X, columns=[f'pixel_{i}' for i in range(X.shape[1])])
    y_df = pd.DataFrame(y, columns=['label'])

    num_features = X_df.shape[1]  # Number of features (pixels)
    num_classes = y_df['label'].nunique()  # Number of unique classes (digits 0-9)

    return X_df, y_df, num_features, num_classes


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


def get_adults_dataset():
    adult = fetch_ucirepo(id=2)

    # data (as pandas dataframes)
    X = adult.data.features
    y = adult.data.targets

    # Drop rows with missing values
    X = X.dropna()
    y = y.loc[X.index]
    # print('Shape after dropping rows with missing values:', X.shape)

    # Convert categorical columns to numeric using one-hot encoding
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Optional: Standardize the numeric features
    scaler = StandardScaler()
    X_encoded_scaled = scaler.fit_transform(X_encoded)
    X_encoded_scaled = pd.DataFrame(X_encoded_scaled, index=X_encoded.index, columns=X_encoded.columns)

    y_encoded = y['income'].apply(lambda x: 1 if '>50K' in x else 0)

    num_features = X_encoded_scaled.shape[1]
    num_classes = y_encoded.nunique()

    return X_encoded_scaled, y_encoded, num_features, num_classes


def get_texas_100_dataset(path='texas100.npz', limit_rows=None):
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


"""
Data preprocessing
"""
# load the dataset
set_random_seed(42)

# X, y, num_features, num_classes = get_mnist_dataset()
# X, y, num_features, num_classes = get_cifar10_dataset()
# X, y, num_features, num_classes = get_adults_dataset()
# X, y, num_features, num_classes = get_purchase_dataset(dataset_path='data/dataset_purchase.csv', keep_rows=40_000)
X, y, num_features, num_classes = get_MUFAC_dataset("./data/mufac-128/custom_train_dataset.csv", "./data/mufac-128/train_images", percentage_of_rows_to_drop = 0.4)
# X, y, num_features, num_classes = get_texas_100_dataset(path='texas100.npz', limit_rows=40_000)


"""
Split data into Target training data, Shadow training data and data as Non Member to test the Attack model
"""

# Split the data into 80% temp and 20% test
X_temp, X_test_MIA, y_temp, y_test_MIA = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Now, split the remaining 80% data into 40% target train and 40% shadow train
X_target, X_shadow, y_target, y_shadow = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
print(X.shape)
print(X_target.shape)

"""
Train Target Model
"""
set_random_seed(42)

features = X_target
labels = y_target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Keep this to create a test dataset for the attack model
X_target_train_set = X_train
y_target_train_set = y_train

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).squeeze()
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).squeeze()

# Hyperparameters for images
hidden_layer_size = 512
learning_rate = 0.001
num_epochs = 30
batch_size = 32
early_stopping_patience = 3
enable_early_stopping = True 

# for tabular data
# hidden_layer_size = 128
# learning_rate = 0.01
# num_epochs = 100
# batch_size = 32
# early_stopping_patience = 3
# enable_early_stopping = False

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_size = num_features
output_size = num_classes
print("NN input size", input_size)
print("NN output size", output_size)

target_model = TargetModel(input_size=input_size, hidden_layer_size=hidden_layer_size, output_size=output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(target_model.parameters(), lr=learning_rate, momentum=0.9)

train_model(target_model, train_loader, test_loader, criterion, optimizer, num_epochs, early_stopping_patience, enable_early_stopping)

train_loss, train_accuracy, train_precision, train_recall = evaluate_model_performance( target_model, criterion, X_train_tensor, y_train_tensor)
test_loss, test_accuracy, test_precision, test_recall = evaluate_model_performance( target_model, criterion, X_test_tensor, y_test_tensor)

print(f'\nTraining Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')
print(f'Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')
print(f'Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}')


"""
Train shadow models
"""
# Hyperparameters for training the shadow model
# hidden_layer_size = 512
# learning_rate = 0.001
# num_epochs = 30
# batch_size = 32
# early_stopping_patience = 3
# enable_early_stopping = False
num_shadow_models = 5


# DataFrame to store outputs from shadow models
shadow_model_outputs_df = pd.DataFrame()

# Prepare features and labels
features = X_shadow
labels = y_shadow

for model_index in tqdm(range(num_shadow_models), desc='Training Shadow Models'):

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).squeeze()
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).squeeze()

    # Create DataLoader instances for training and testing
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    input_size = num_features
    output_size = num_classes

    shadow_model = ShadowModel(input_size, hidden_layer_size, output_size)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(shadow_model.parameters(), lr=learning_rate, momentum=0.9)

    # Train the shadow model
    train_model(shadow_model, train_loader, test_loader, criterion, optimizer, num_epochs, early_stopping_patience, enable_early_stopping)

    print(f'\nGenerating data from shadow model ({model_index + 1}) outputs\n')

    # Sample equal number of members and non-members
    sample_size = min(len(X_train), len(X_test))
    X_member_sample = X_train.sample(n=sample_size, replace=False)
    X_non_member_sample = X_test.sample(n=sample_size, replace=False)

    # Generate dataset with membership status from current shadow model
    current_shadow_model_outputs_df = create_membership_dataframe(shadow_model, X_member_sample, X_non_member_sample)

    # Append to the results DataFrame
    shadow_model_outputs_df = pd.concat([shadow_model_outputs_df, current_shadow_model_outputs_df], ignore_index=True)

print('\nResulting shadow model outputs dataset shape:', shadow_model_outputs_df.shape)

"""
Train the attack model using the shadow models outputs
"""

attack_model = train_attack_model_on_output_data(shadow_model_outputs_df, 42, test_model_capabilities=True)


""" ##### MIA ##### """

"""
Based on the Target Model predictions, use the Attack Model to predict membership status for each data point
"""

# Sample an equal number of non-member data
sample_size = min(len(X_test_MIA), len(X_target_train_set))
train_member_data = X_target_train_set.sample(n=sample_size, replace=False)
train_non_member_data = X_test_MIA.sample(n=sample_size, replace=False)

# Create a dataset by QUERYING the TARGET model on member and non-member data
dataset_from_target_model_outputs = create_membership_dataframe(target_model, train_member_data, train_non_member_data)
print('testing MIA for the following dataset: ', dataset_from_target_model_outputs.shape)

# Display relevant information
print('Target model outputs dataset shape:', dataset_from_target_model_outputs.shape)

# Use the ATTACK model to predict membership status
evaluate_attack_model(dataset_from_target_model_outputs, attack_model, 'ROC Curve - Test set', plot_choice='roc')

# Get a typical input tensor from the training data loader

# Save the models 
torch.save(target_model, 'models/dataset_target_model.pth')
joblib.dump(attack_model , 'models/dataset_attack_model.jolib')