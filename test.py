import cupy as cp

def test(model, test_data, test_labels, batch_size=128):
    """
    Evaluate the model on the test set.
    
    Args:
        model: ConvNet model
        test_data: Test images
        test_labels: Test labels
        batch_size: Batch size for evaluation
    
    Returns:
        test_acc: Test accuracy
    """
    test_acc = 0.0
    num_batches = 0
    
    for i in range(0, len(test_data), batch_size):
        x_batch = test_data[i:i+batch_size]
        y_batch = test_labels[i:i+batch_size]
        y_pred = model.forward(x_batch, training=False, return_features=False)
        test_acc += cp.mean(cp.argmax(y_pred, axis=1) == y_batch)
        num_batches += 1
    
    test_acc /= num_batches
    print(f"测试准确率: {test_acc:.4f}")
    return float(test_acc.get())