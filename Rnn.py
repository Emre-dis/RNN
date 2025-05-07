import numpy as np
import matplotlib.pyplot as plt
import re

# Import data directly from data.py
from data import train_data, test_data


class RNN:
    # A many-to-one Vanilla Recurrent Neural Network for sentiment analysis

    def __init__(self, vocab_size, output_size, hidden_size=64, bptt_truncate=4):
        # Store dimensions
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bptt_truncate = bptt_truncate  # Limit for backpropagation through time

        # Weights initialization with Xavier/Glorot initialization
        self.Wxh = np.random.randn(hidden_size, vocab_size) * np.sqrt(2.0 / (vocab_size + hidden_size))
        self.Whh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / (hidden_size + hidden_size))
        self.Why = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / (hidden_size + output_size))

        # Biases initialization
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

        # Memory variables for AdaGrad
        self.mWxh, self.mWhh, self.mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        self.mbh, self.mby = np.zeros_like(self.bh), np.zeros_like(self.by)

    def forward(self, inputs):
        '''
        Perform a forward pass of the RNN using the given inputs.
        Returns the final output and hidden state.
        - inputs is an array of one-hot vectors with shape (seq_length, vocab_size, 1).
        '''
        # Initialize hidden state as zeros
        h = np.zeros((self.hidden_size, 1))

        # Store inputs and hidden states for backprop
        self.last_inputs = inputs
        self.last_hs = {0: h}

        # Perform each step of the RNN
        for i, x in enumerate(inputs):
            # Calculate hidden state
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hs[i + 1] = h

        # Compute the output (logits)
        y = self.Why @ h + self.by

        # Apply sigmoid activation for binary classification
        p = 1 / (1 + np.exp(-y))

        return p, h

    def backprop(self, d_y, learn_rate=2e-2, decay_rate=0.95):
        '''
        Perform a backward pass of the RNN with truncated BPTT.
        - d_y (dL/dy) has shape (output_size, 1).
        - learn_rate is a float.
        '''
        n = len(self.last_inputs)

        # Calculate dL/dWhy and dL/dby
        d_Why = d_y @ self.last_hs[n].T
        d_by = d_y.copy()

        # Initialize dL/dWhh, dL/dWxh, and dL/dbh to zero
        d_Whh = np.zeros_like(self.Whh)
        d_Wxh = np.zeros_like(self.Wxh)
        d_bh = np.zeros_like(self.bh)

        # Backpropagate through time, but limit steps to prevent vanishing/exploding gradients
        d_h = self.Why.T @ d_y

        # Use truncated backpropagation through time
        for t in range(n, max(0, n - self.bptt_truncate - 1), -1):
            if t == 0:
                break

            # Calculate gradient at current timestep
            # tanh derivative: 1 - tanh²(x)
            temp = (1 - self.last_hs[t] ** 2) * d_h

            # Update gradients
            d_bh += temp
            d_Wxh += temp @ self.last_inputs[t - 1].T
            d_Whh += temp @ self.last_hs[t - 1].T

            # Propagate error back to previous timestep
            d_h = self.Whh.T @ temp

        # Clip gradients to prevent exploding gradients
        for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
            np.clip(d, -5, 5, out=d)

        # Update weights and biases using AdaGrad
        for param, d_param, mem in zip(
                [self.Wxh, self.Whh, self.Why, self.bh, self.by],
                [d_Wxh, d_Whh, d_Why, d_bh, d_by],
                [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]
        ):
            mem += d_param * d_param  # Update memory
            param -= learn_rate * d_param / np.sqrt(mem + 1e-8)  # Update parameter with adaptive learning rate


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


def process_data(data_dict, vocab):
    """
    Process data and convert to one-hot vectors
    """
    processed_data = []

    for text, sentiment in data_dict.items():
        # Tokenize text
        tokens = tokenize(text)

        # Create one-hot encoded vectors
        one_hot_vectors = []
        for token in tokens:
            # Create a zero vector with size of vocabulary
            one_hot = np.zeros((len(vocab), 1))

            # Set the index of the token to 1 if it's in vocabulary
            if token in vocab:
                one_hot[vocab[token]] = 1
            one_hot_vectors.append(one_hot)

        # Create the target vector (1 for positive sentiment, 0 for negative)
        target = np.array([[1.0]]) if sentiment else np.array([[0.0]])

        processed_data.append((one_hot_vectors, target))

    return processed_data


def binary_cross_entropy_loss(predictions, targets):
    """
    Calculate binary cross-entropy loss with numerical stability.
    """
    epsilon = 1e-12  # Small value to avoid log(0)
    predictions = np.clip(predictions, epsilon, 1 - epsilon)  # Clip values to avoid numerical instability
    loss = -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    return loss


def train_rnn(model, data, epochs=1000, learning_rate=0.01, batch_size=16):
    """
    Train the RNN model on the given data with batching and learning rate decay.
    """
    losses = []
    accuracies = []
    initial_learning_rate = learning_rate

    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0

        # Calculate decayed learning rate
        current_lr = initial_learning_rate * (0.95 ** (epoch // 100))

        # Shuffle data for each epoch
        np.random.shuffle(data)

        # Process data in batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_loss = 0
            batch_correct = 0

            for inputs, target in batch:
                # Forward pass
                prediction, _ = model.forward(inputs)

                # Compute loss
                loss = binary_cross_entropy_loss(prediction, target)
                batch_loss += loss

                # Check if prediction is correct
                predicted_class = 1 if prediction >= 0.5 else 0
                if predicted_class == target[0][0]:
                    batch_correct += 1

                # Compute derivative of loss with respect to output
                # For binary cross-entropy with sigmoid:
                # dL/dy = prediction - target
                d_L_d_y = prediction - target

                # Backpropagation
                model.backprop(d_L_d_y, current_lr)

            total_loss += batch_loss
            correct_predictions += batch_correct

        # Calculate average loss and accuracy for this epoch
        avg_loss = total_loss / len(data)
        accuracy = correct_predictions / len(data)

        # Store metrics for plotting
        losses.append(avg_loss)
        accuracies.append(accuracy)

        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, LR: {current_lr:.6f}")

    return losses, accuracies


def evaluate_rnn(model, data):
    """
    Evaluate the RNN model on the given data.
    """
    correct_predictions = 0
    predictions = []
    actual = []
    total_loss = 0

    for inputs, target in data:
        # Forward pass
        prediction, _ = model.forward(inputs)

        # Compute loss
        loss = binary_cross_entropy_loss(prediction, target)
        total_loss += loss

        # Convert prediction to class
        predicted_class = 1 if prediction >= 0.5 else 0
        predictions.append(predicted_class)
        actual.append(int(target[0][0]))

        # Check if prediction is correct
        if predicted_class == target[0][0]:
            correct_predictions += 1

    accuracy = correct_predictions / len(data)
    avg_loss = total_loss / len(data)

    # Compute confusion matrix
    true_positive = sum(1 for p, a in zip(predictions, actual) if p == 1 and a == 1)
    true_negative = sum(1 for p, a in zip(predictions, actual) if p == 0 and a == 0)
    false_positive = sum(1 for p, a in zip(predictions, actual) if p == 1 and a == 0)
    false_negative = sum(1 for p, a in zip(predictions, actual) if p == 0 and a == 1)

    confusion_matrix = np.array([
        [true_negative, false_positive],
        [false_negative, true_positive]
    ])

    return accuracy, avg_loss, confusion_matrix, predictions


def plot_results(train_losses, train_accuracies, test_accuracy, test_loss):
    """
    Plot the training metrics and test results.
    """
    plt.figure(figsize=(15, 5))

    # Plot training loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Plot training accuracy
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    # Plot test results
    plt.subplot(1, 3, 3)
    bars = plt.bar(['Test Accuracy', 'Test Loss'], [test_accuracy, test_loss])
    bars[0].set_color('green')
    bars[1].set_color('red')
    plt.title('Test Results')
    plt.ylabel('Value')

    plt.tight_layout()
    plt.savefig('improved_rnn_results.png')
    plt.show()


def print_confusion_matrix(confusion_matrix):
    """
    Print the confusion matrix in a readable format.
    """
    print("Confusion Matrix:")
    print("                 Predicted")
    print("                 Negative  Positive")
    print(f"Actual Negative    {confusion_matrix[0, 0]}        {confusion_matrix[0, 1]}")
    print(f"      Positive     {confusion_matrix[1, 0]}        {confusion_matrix[1, 1]}")

    # Calculate precision, recall, and F1 score
    true_positive = confusion_matrix[1, 1]
    false_positive = confusion_matrix[0, 1]
    false_negative = confusion_matrix[1, 0]
    true_negative = confusion_matrix[0, 0]

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")


def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Build vocabulary from training and test data
    vocabulary = build_vocabulary({**train_data, **test_data})
    vocab_size = len(vocabulary)

    print(f"Vocabulary size: {vocab_size}")

    # Process the data
    train_processed = process_data(train_data, vocabulary)
    test_processed = process_data(test_data, vocabulary)

    print(f"Training data size: {len(train_processed)}")
    print(f"Test data size: {len(test_processed)}")

    # Check class distribution
    positive_count = sum(1 for _, target in train_processed if target[0][0] == 1)
    negative_count = len(train_processed) - positive_count
    print(f"Training data class distribution: Positive {positive_count}, Negative {negative_count}")

    # Initialize the RNN model
    output_size = 1  # Binary classification
    hidden_size = 100  # Optimized hidden size

    print(f"Input size: {vocab_size}")
    print(f"Output size: {output_size}")
    print(f"Hidden size: {hidden_size}")

    model = RNN(vocab_size, output_size, hidden_size, bptt_truncate=4)

    # Train the model with optimized hyperparameters
    print("\nTraining the model...")
    train_losses, train_accuracies = train_rnn(
        model,
        train_processed,
        epochs=1500,  # More epochs for better convergence
        learning_rate=0.01,  # Starting learning rate
        batch_size=8  # Small batch size for this dataset
    )

    # Evaluate the model
    print("\nEvaluating the model...")
    test_accuracy, test_loss, confusion_matrix, predictions = evaluate_rnn(model, test_processed)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print_confusion_matrix(confusion_matrix)

    # Print some example predictions
    print("\nSample Predictions:")
    sample_size = min(5, len(test_processed))
    for i in range(sample_size):
        _, target = test_processed[i]
        text = list(test_data.keys())[i]
        pred_class = predictions[i]
        actual_class = int(target[0][0])
        correct = "✓" if pred_class == actual_class else "✗"
        print(f"{correct} Text: '{text}', Predicted: {pred_class}, Actual: {actual_class}")

    # Plot results
    plot_results(train_losses, train_accuracies, test_accuracy, test_loss)


if __name__ == "__main__":
    main()