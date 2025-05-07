import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import re
from sklearn.metrics import confusion_matrix

# Import data directly from data.py
from data import train_data, test_data


def tokenize(text):
    """
    Tokenize text into words
    """
    text = text.lower().strip()
    return re.findall(r'\w+', text)


def build_vocabulary(data_dict):
    """
    Build vocabulary from the dataset
    """
    vocab = {'<PAD>': 0}  # 0 for padding
    word_id = 1

    for sentence in data_dict.keys():
        for word in tokenize(sentence):
            if word not in vocab:
                vocab[word] = word_id
                word_id += 1

    return vocab


def prepare_data(data_dict, vocab):
    """
    Prepare data for PyTorch model
    """
    X = []
    y = []

    for sentence, sentiment in data_dict.items():
        # Convert sentence to token indices
        indices = [vocab.get(word, 0) for word in tokenize(sentence)]
        X.append(indices)
        # Convert True/False to 1/0
        y.append(1 if sentiment else 0)

    return X, y


class SentimentDataset(Dataset):
    """PyTorch dataset class for sentiment analysis"""

    def __init__(self, X, y, max_length=20):
        self.X = X
        self.y = y
        self.max_length = max_length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Pad sequence to fixed length
        sentence = self.X[idx]
        padded = self._pad_sequence(sentence)

        # Convert to tensors
        x_tensor = torch.LongTensor(padded)
        y_tensor = torch.LongTensor([self.y[idx]])

        return x_tensor, y_tensor

    def _pad_sequence(self, sequence):
        """Pad or truncate sequence to max_length"""
        if len(sequence) >= self.max_length:
            return sequence[:self.max_length]
        else:
            return sequence + [0] * (self.max_length - len(sequence))


class GRUModel(nn.Module):
    """PyTorch GRU model for sentiment analysis"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout=0.3):
        super(GRUModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # GRU layer - an advanced version of RNN
        self.gru = nn.GRU(embedding_dim,
                          hidden_dim,
                          batch_first=True,
                          dropout=dropout)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text: [batch size, sequence length]

        # Embedding: [batch size, sequence length, embedding dim]
        embedded = self.embedding(text)

        # GRU: output: [batch size, sequence length, hidden dim]
        #      hidden: [1, batch size, hidden dim]
        output, hidden = self.gru(embedded)

        # Get the last hidden state
        # hidden: [batch size, hidden dim]
        hidden = hidden.squeeze(0)

        # Apply dropout
        hidden = self.dropout(hidden)

        # Output layer: [batch size, output dim]
        output = self.fc(hidden)

        return output


def train_model(model, train_loader, optimizer, criterion, device):
    """Train PyTorch model for one epoch"""
    model.train()
    epoch_loss = 0

    for batch_idx, (text, labels) in enumerate(train_loader):
        # Move data to device
        text, labels = text.to(device), labels.squeeze(1).to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(text)

        # Calculate loss
        loss = criterion(predictions, labels)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


def evaluate_model(model, test_loader, criterion, device):
    """Evaluate PyTorch model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for text, labels in test_loader:
            # Move data to device
            text, labels = text.to(device), labels.squeeze(1).to(device)

            # Forward pass
            predictions = model(text)

            # Calculate loss
            loss = criterion(predictions, labels)
            total_loss += loss.item()

            # Calculate predictions
            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store predictions and true labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    return total_loss / len(test_loader), accuracy, all_preds, all_labels


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Build vocabulary from training and test data
    vocab = build_vocabulary({**train_data, **test_data})
    vocab_size = len(vocab)

    print(f"Vocabulary size: {vocab_size}")

    # Prepare data
    X_train, y_train = prepare_data(train_data, vocab)
    X_test, y_test = prepare_data(test_data, vocab)

    # Model hyperparameters
    embedding_dim = 64
    hidden_dim = 128
    output_dim = 2  # Binary classification (0 or 1)
    batch_size = 32
    epochs = 50
    learning_rate = 0.001
    dropout = 0.3

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets
    train_dataset = SentimentDataset(X_train, y_train)
    test_dataset = SentimentDataset(X_test, y_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Create model
    model = GRUModel(vocab_size, embedding_dim, hidden_dim, output_dim, dropout)
    model = model.to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training
    print("Starting PyTorch GRU model training...")
    train_losses = []
    test_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        # Train model
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        # Evaluate model
        test_loss, test_accuracy, _, _ = evaluate_model(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

    # Final evaluation
    _, final_accuracy, all_preds, all_labels = evaluate_model(model, test_loader, criterion, device)

    print(f"\nTest accuracy: {final_accuracy:.4f}")

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # True Positive, False Positive, True Negative, False Negative
    tn, fp, fn, tp = cm.ravel()

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(f"              Predicted Positive  Predicted Negative")
    print(f"Actual Positive     {tp}                {fn}")
    print(f"Actual Negative     {fp}                {tn}")

    # Other metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    # Visualize training and test losses
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('pytorch_gru_results.png')
    plt.show()


if __name__ == "__main__":
    main()