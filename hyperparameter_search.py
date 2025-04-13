import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from model import ConvNet
from train import load_cifar10_data, train
from test import test

def hyperparameter_search(data_dir, output_dir, model_dir):
    """
    Perform hyperparameter search for ConvNet on CIFAR-10 dataset.
    
    Args:
        data_dir: Directory containing CIFAR-10 batch files
        output_dir: Directory to save plots and results
        model_dir: Directory to save model checkpoints
    
    Returns:
        best_params: Best hyperparameter combination
        best_dropout_rate: Best dropout rate
        best_val_acc: Best validation accuracy
        test_acc: Test accuracy of best model
    """
    train_data, train_labels, val_data, val_labels, test_data, test_labels = load_cifar10_data(data_dir)
    
    lrs = [0.1, 0.05, 0.01]
    regs = [0.01, 0.005, 0.001]
    filter_sizes = [(32, 64), (32, 32), (16, 32)]
    
    best_val_acc = 0.0
    best_params = None
    best_model_file = None
    results = []
    
    id = 0
    for lr in lrs:
        for reg in regs:
            for num_filters1, num_filters2 in filter_sizes:
                id += 1
                print(f"\nHyperparameter combination {id}: lr={lr}, reg={reg}, num_filters1={num_filters1}, num_filters2={num_filters2}")
                model = ConvNet(num_filters1=num_filters1, num_filters2=num_filters2, dropout_rate=0.5)
                model_file = os.path.join(model_dir, f'model_{id}.npz')
                
                train_losses, train_accuracies, val_losses, val_accuracies = train(
                    model, train_data, train_labels, val_data, val_labels, id,
                    lr=lr, reg=reg, epochs=20, batch_size=128, patience=5, model_file=model_file
                )
                
                val_acc = max(val_accuracies)
                print(f"Hyperparameter combination {id}: lr={lr}, reg={reg}, num_filters1={num_filters1}, num_filters2={num_filters2}, 最佳验证准确率: {val_acc:.4f}")
                
                results.append({
                    'id': id,
                    'lr': lr,
                    'reg': reg,
                    'num_filters1': num_filters1,
                    'num_filters2': num_filters2,
                    'val_acc': val_acc,
                    'model_file': model_file,
                    'train_losses': train_losses,
                    'val_losses': val_losses
                })
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_params = {'lr': lr, 'reg': reg, 'num_filters1': num_filters1, 'num_filters2': num_filters2}
                    best_model_file = model_file
                
                plt.figure(figsize=(12, 4))
    
                plt.subplot(1, 2, 1)
                plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
                plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'Loss: lr={lr}, reg={reg}, filters=({num_filters1},{num_filters2})')
                plt.legend()
                plt.tight_layout()

                plt.subplot(1, 2, 2)
                plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Val Accuracy', color='green')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title('Validation Accuracy')
                plt.legend()

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'loss_plot_{id}.png'))
                plt.close()
    
    results_df = pd.DataFrame([
        {
            'ID': res['id'],
            'Learning Rate': res['lr'],
            'Regularization': res['reg'],
            'Num Filters1': res['num_filters1'],
            'Num Filters2': res['num_filters2'],
            'Best Val Acc': res['val_acc']
        }
        for res in results
    ])
    results_df.to_csv(os.path.join(output_dir, 'hyperparameter_results.csv'), index=False)
    print(f"\nAll hyperparameter combination has saved to: {os.path.join(output_dir, 'hyperparameter_results.csv')}")
    
    print("\nResults of all hyperparameter combination:")
    for res in results:
        print(f"Combinations: {res['id']}: lr={res['lr']}, reg={res['reg']}, num_filters1={res['num_filters1']}, num_filters2={res['num_filters2']}, 验证准确率: {res['val_acc']:.4f}")
    
    dropout_rates = [0, 0.3, 0.5, 0.7]
    dropout_results = []
    best_dropout_acc = 0.0
    best_dropout_rate = None
    
    print(f"\nUsing the best combination to search dropout rate: lr={best_params['lr']}, reg={best_params['reg']}, num_filters1={best_params['num_filters1']}, num_filters2={best_params['num_filters2']}")
    
    for dropout_rate in dropout_rates:
        print(f"\nDropout rate: {dropout_rate}")
        model = ConvNet(
            num_filters1=best_params['num_filters1'],
            num_filters2=best_params['num_filters2'],
            dropout_rate=dropout_rate
        )
        model_file = os.path.join(model_dir, f'model_dropout_{dropout_rate}.npz')
        
        train_losses, train_accuracies, val_losses, val_accuracies = train(
            model, train_data, train_labels, val_data, val_labels, f"dropout_{dropout_rate}",
            lr=best_params['lr'], reg=best_params['reg'], epochs=20, batch_size=128, patience=5, model_file=model_file
        )
        
        val_acc = max(val_accuracies)
        print(f"Dropout rate {dropout_rate}, Best validation accuracy: {val_acc:.4f}")
        
        dropout_results.append({
            'dropout_rate': dropout_rate,
            'val_acc': val_acc
        })
        
        if val_acc > best_dropout_acc:
            best_dropout_acc = val_acc
            best_dropout_rate = dropout_rate
    
    dropout_df = pd.DataFrame([
        {
            'Dropout Rate': res['dropout_rate'],
            'Best Val Acc': res['val_acc']
        }
        for res in dropout_results
    ])
    dropout_df.to_csv(os.path.join(output_dir, 'dropout_results.csv'), index=False)
    print(f"\nDropout rate table has saved to: {os.path.join(output_dir, 'dropout_results.csv')}")
    
    print(f"\nBest dropout rate: {best_dropout_rate}, 验证准确率: {best_dropout_acc:.4f}")
    
    print(f"\nUsing the best combination to test on test set.")
    model = ConvNet(
        num_filters1=best_params['num_filters1'],
        num_filters2=best_params['num_filters2'],
        dropout_rate=best_dropout_rate
    )
    
    best_dropout_model_file = os.path.join(model_dir, f'model_dropout_{best_dropout_rate}.npz')
    weights = cp.load(best_dropout_model_file)
    model.w1, model.b1 = weights['w1'], weights['b1']
    model.w2, model.b2 = weights['w2'], weights['b2']
    model.w3, model.b3 = weights['w3'], weights['b3']
    model.gamma1, model.beta1 = weights['gamma1'], weights['beta1']
    model.gamma2, model.beta2 = weights['gamma2'], weights['beta2']
    model.bn_cache1['running_mean'] = weights['running_mean1']
    model.bn_cache1['running_var'] = weights['running_var1']
    model.bn_cache2['running_mean'] = weights['running_mean2']
    model.bn_cache2['running_var'] = weights['running_var2']
    
    test_acc = test(model, test_data, test_labels)
    print(f"The best test accuracy: {test_acc:.4f}")
    
    return best_params, best_dropout_rate, best_val_acc, test_acc

def main():
    """Parse command-line arguments and run hyperparameter search."""
    parser = argparse.ArgumentParser(description="Hyperparameter search for ConvNet on CIFAR-10.")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing CIFAR-10 batch files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save plots and results')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory to save model checkpoints')
    args = parser.parse_args()

    best_params, best_dropout_rate, best_val_acc, test_acc = hyperparameter_search(
        args.data_dir, args.output_dir, args.model_dir
    )

if __name__ == "__main__":
    main()