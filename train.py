import cupy as cp
import pickle
import os
import sys
from tqdm import tqdm
from model import ConvNet

def load_cifar10_batch(file):
    """
    Load a single CIFAR-10 batch from file.
    
    Args:
        file: Path to batch file
    
    Returns:
        data: Image data
        labels: Corresponding labels
    """
    with open(file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = cp.array(batch[b'data'].reshape(-1, 3, 32, 32) / 255.0)
    labels = cp.array(batch[b'labels'])
    return data, labels

def load_cifar10_data(data_dir):
    """
    Load and preprocess CIFAR-10 dataset.
    
    Args:
        data_dir: Directory containing CIFAR-10 batch files
    
    Returns:
        train_data: Training images
        train_labels: Training labels
        val_data: Validation images
        val_labels: Validation labels
        test_data: Test images
        test_labels: Test labels
    """
    train_data, train_labels = [], []
    for i in range(1, 6):
        file = os.path.join(data_dir, f'data_batch_{i}')
        data, labels = load_cifar10_batch(file)
        train_data.append(data)
        train_labels.append(labels)

    train_data = cp.concatenate(train_data)
    train_labels = cp.concatenate(train_labels)

    num_total = len(train_data)
    indices = cp.random.permutation(num_total)
    train_data = train_data[indices]
    train_labels = train_labels[indices]

    n_train = int(0.8 * len(train_data))

    val_data = train_data[n_train:]
    val_labels = train_labels[n_train:]

    train_data = train_data[:n_train]
    train_labels = train_labels[:n_train]

    test_file = os.path.join(data_dir, 'test_batch')
    test_data, test_labels = load_cifar10_batch(test_file)
    test_num_total = len(test_data)
    test_indices = cp.random.permutation(test_num_total)
    test_data = test_data[test_indices]
    test_labels = test_labels[test_indices]

    return train_data, train_labels, val_data, val_labels, test_data, test_labels

def compute_validation_metrics(model, val_data, val_labels, reg, batch_size=128):
    """
    Compute loss and accuracy on validation set.
    
    Args:
        model: ConvNet model
        val_data: Validation images
        val_labels: Validation labels
        reg: Regularization strength
        batch_size: Batch size for evaluation
    
    Returns:
        val_loss: Average validation loss
        val_acc: Average validation accuracy
    """
    val_loss = 0.0
    val_acc = 0.0
    num_batches = 0
    
    for i in range(0, len(val_data), batch_size):
        x_batch = val_data[i:i+batch_size]
        y_batch = val_labels[i:i+batch_size]
        y_pred = model.forward(x_batch, training=False, return_features=False)
        val_loss += -cp.mean(cp.sum(cp.eye(10)[y_batch] * cp.log(y_pred + 1e-8), axis=1))
        val_acc += cp.mean(cp.argmax(y_pred, axis=1) == y_batch)
        num_batches += 1
    
    val_loss /= num_batches
    val_loss += 0.5 * reg * (cp.sum(model.w1**2) + cp.sum(model.w2**2) + cp.sum(model.w3**2))
    val_acc /= num_batches
    return float(val_loss.get()), float(val_acc.get())

def train(model, train_data, train_labels, val_data, val_labels, ID, lr=0.01, reg=0.001, epochs=10, batch_size=128, patience=5, model_file=None):
    """
    Train the ConvNet model.
    
    Args:
        model: ConvNet model
        train_data: Training images
        train_labels: Training labels
        val_data: Validation images
        val_labels: Validation labels
        ID: Identifier for the training run
        lr: Initial learning rate
        reg: Regularization strength
        epochs: Number of epochs
        batch_size: Batch size
        patience: Patience for early stopping
        model_file: Path to save model weights
    
    Returns:
        train_losses: List of training losses per epoch
        train_accuracies: List of training accuracies per epoch
        val_losses: List of validation losses per epoch
        val_accuracies: List of validation accuracies per epoch
    """
    best_acc = 0
    best_epoch = 0
    n = len(train_data)
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    patience_counter = 0
    
    for epoch in range(epochs):
        lr_epoch = lr * (0.995 ** epoch)
        perm = cp.random.permutation(n)
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        num_batches = 0
        
        for i in tqdm(range(0, n, batch_size), desc=f"Epoch {epoch+1}/{epochs}", file=sys.stdout, leave=False):
            batch_idx = perm[i:i+batch_size]
            x_batch = train_data[batch_idx]
            y_batch = cp.eye(10)[train_labels[batch_idx]]
            
            y_pred = model.forward(x_batch, training=True, return_features=False)
            loss = -cp.mean(cp.sum(y_batch * cp.log(y_pred + 1e-8), axis=1))
            loss += 0.5 * reg * (cp.sum(model.w1**2) + cp.sum(model.w2**2) + cp.sum(model.w3**2))
            epoch_train_loss += float(loss.get())
            num_batches += 1

            batch_acc = cp.mean(cp.argmax(y_pred, axis=1) == train_labels[batch_idx])
            epoch_train_acc += float(batch_acc.get())
            
            grads = model.backward(x_batch, y_batch, reg)
            model.update(grads, lr_epoch)
        
        epoch_train_loss /= num_batches
        epoch_train_acc /= num_batches
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        
        val_loss, val_acc = compute_validation_metrics(model, val_data, val_labels, reg, batch_size)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")
        
        if val_accuracies[-1] > best_acc:
            best_acc = val_accuracies[-1]
            best_epoch = epoch + 1
            patience_counter = 0
            if model_file:
                cp.savez(model_file, 
                         w1=model.w1, b1=model.b1, 
                         w2=model.w2, b2=model.b2,
                         w3=model.w3, b3=model.b3, 
                         gamma1=model.gamma1, beta1=model.beta1, 
                         gamma2=model.gamma2, beta2=model.beta2,
                         running_mean1=model.bn_cache1.get('running_mean', cp.zeros_like(model.gamma1)),
                         running_var1=model.bn_cache1.get('running_var', cp.ones_like(model.gamma1)),
                         running_mean2=model.bn_cache2.get('running_mean', cp.zeros_like(model.gamma2)),
                         running_var2=model.bn_cache2.get('running_var', cp.ones_like(model.gamma2)))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early exit at {epoch+1}th epoch, best validation accuracy: {best_acc:.4f} 在第 {best_epoch} 轮")
                break

        print("Trying to free CUDA memory...")
        cp.get_default_memory_pool().free_all_blocks()
    
    print(f"Predicted distribution: {cp.bincount(cp.argmax(y_pred, axis=1), minlength=10).get()}")

    return train_losses, train_accuracies, val_losses, val_accuracies