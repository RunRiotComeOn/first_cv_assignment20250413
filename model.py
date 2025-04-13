import cupy as cp

def relu(x):
    """Apply ReLU activation function."""
    return cp.maximum(0, x)

def relu_deriv(x):
    """Compute derivative of ReLU activation function."""
    return x > 0

def softmax(x):
    """Apply softmax activation function."""
    exp_x = cp.exp(x - cp.max(x, axis=1, keepdims=True))
    return exp_x / cp.sum(exp_x, axis=1, keepdims=True)

def conv_forward(x, w, b, stride=1, pad=2):
    """
    Perform forward pass of convolution operation.
    
    Args:
        x: Input data of shape (N, C, H, W)
        w: Filter weights of shape (F, C, FH, FW)
        b: Biases of shape (F,)
        stride: Stride of convolution
        pad: Padding size
    
    Returns:
        out: Output data of shape (N, F, out_height, out_width)
    """
    n, c, height, width = x.shape
    f, _, fh, fw = w.shape
    out_height = (height + 2 * pad - fh) // stride + 1
    out_width = (width + 2 * pad - fw) // stride + 1
    out = cp.zeros((n, f, out_height, out_width), dtype=cp.float64)

    if pad > 0:
        x_padded = cp.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    else:
        x_padded = x

    for i in range(out_height):
        for j in range(out_width):
            h_start = i * stride
            h_end = h_start + fh
            w_start = j * stride
            w_end = w_start + fw
            window = x_padded[:, :, h_start:h_end, w_start:w_end]
            out[:, :, i, j] = cp.tensordot(window, w, axes=([1, 2, 3], [1, 2, 3])) + b

    return out

def conv_backward(dout, x, w, stride=1, pad=2):
    """
    Perform backward pass of convolution operation.
    
    Args:
        dout: Gradient of loss w.r.t. output of shape (N, F, out_height, out_width)
        x: Input data of shape (N, C, H, W)
        w: Filter weights of shape (F, C, FH, FW)
        stride: Stride of convolution
        pad: Padding size
    
    Returns:
        dx: Gradient w.r.t. input
        dw: Gradient w.r.t. weights
        db: Gradient w.r.t. biases
    """
    n, c, height, width = x.shape
    f, _, fh, fw = w.shape
    _, _, out_height, out_width = dout.shape

    dx = cp.zeros_like(x)
    dw = cp.zeros_like(w)
    db = cp.sum(dout, axis=(0, 2, 3))

    if pad > 0:
        x_padded = cp.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        dx_padded = cp.zeros_like(x_padded)
    else:
        x_padded = x
        dx_padded = dx

    for i in range(out_height):
        for j in range(out_width):
            h_start = i * stride
            h_end = h_start + fh
            w_start = j * stride
            w_end = w_start + fw
            window = x_padded[:, :, h_start:h_end, w_start:w_end]
            dout_slice = dout[:, :, i, j][:, :, None, None, None]
            dw += cp.tensordot(dout[:, :, i, j], window, axes=(0, 0))
            dx_padded[:, :, h_start:h_end, w_start:w_end] += cp.tensordot(dout[:, :, i, j], w, axes=(1, 0))

    if pad > 0:
        dx = dx_padded[:, :, pad:-pad, pad:-pad]
    else:
        dx = dx_padded

    return dx, dw, db

def maxpool_forward(x, pool_size=2, stride=2):
    """
    Perform forward pass of max pooling operation.
    
    Args:
        x: Input data of shape (N, C, H, W)
        pool_size: Size of pooling window
        stride: Stride of pooling
    
    Returns:
        out: Pooled output
        mask: Mask indicating max locations
    """
    n, c, height, width = x.shape
    out_height = (height - pool_size) // stride + 1
    out_width = (width - pool_size) // stride + 1
    out = cp.zeros((n, c, out_height, out_width))
    mask = cp.zeros((n, c, height, width))

    for i in range(out_height):
        for j in range(out_width):
            h_start = i * stride
            h_end = h_start + pool_size
            w_start = j * stride
            w_end = w_start + pool_size
            window = x[:, :, h_start:h_end, w_start:w_end]
            out[:, :, i, j] = cp.max(window, axis=(2, 3))
            max_indices = cp.argmax(window.reshape(n, c, -1), axis=2)
            mask_flat = cp.zeros((n, c, pool_size * pool_size))
            mask_flat[cp.arange(n)[:, None], cp.arange(c)[None, :], max_indices] = 1
            mask[:, :, h_start:h_end, w_start:w_end] = mask_flat.reshape(n, c, pool_size, pool_size)

    return out, mask

def maxpool_backward(dout, x, mask, pool_size=2, stride=2):
    """
    Perform backward pass of max pooling operation.
    
    Args:
        dout: Gradient of loss w.r.t. output
        x: Input data
        mask: Mask from forward pass
        pool_size: Size of pooling window
        stride: Stride of pooling
    
    Returns:
        dx: Gradient w.r.t. input
    """
    n, c, height, width = x.shape
    _, _, out_height, out_width = dout.shape
    dx = cp.zeros_like(x)

    for i in range(out_height):
        for j in range(out_width):
            h_start = i * stride
            h_end = h_start + pool_size
            w_start = j * stride
            w_end = w_start + pool_size
            dx[:, :, h_start:h_end, w_start:w_end] += \
                dout[:, :, i, j][:, :, None, None] * mask[:, :, h_start:h_end, w_start:w_end]

    return dx

def fc_forward(x, w, b):
    """Perform forward pass of fully connected layer."""
    return cp.tensordot(x, w, axes=1) + b

def fc_backward(dout, x, w):
    """
    Perform backward pass of fully connected layer.
    
    Args:
        dout: Gradient of loss w.r.t. output
        x: Input data
        w: Weights
    
    Returns:
        dx: Gradient w.r.t. input
        dw: Gradient w.r.t. weights
        db: Gradient w.r.t. biases
    """
    dx = cp.tensordot(dout, w.T, axes=1)
    dw = cp.tensordot(x.T, dout, axes=1)
    db = cp.sum(dout, axis=0)
    return dx, dw, db

def batch_norm_forward(x, gamma, beta, eps=1e-5, momentum=0.9, training=True, cache=None):
    """
    Perform forward pass of batch normalization.
    
    Args:
        x: Input data of shape (N, C, H, W)
        gamma: Scale parameter
        beta: Shift parameter
        eps: Small constant for numerical stability
        momentum: Momentum for running mean and variance
        training: Whether in training mode
        cache: Cache for storing intermediate values
    
    Returns:
        out: Normalized output
    """
    N, C, H, W = x.shape
    x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, C)
    
    if training:
        mu = cp.mean(x_reshaped, axis=0)
        var = cp.var(x_reshaped, axis=0)
        x_norm = (x_reshaped - mu) / cp.sqrt(var + eps)
        out = gamma * x_norm + beta
        
        if cache is not None:
            cache['mu'] = mu
            cache['var'] = var
            cache['x_norm'] = x_norm
            cache['x'] = x_reshaped
            if 'running_mean' not in cache:
                cache['running_mean'] = mu
                cache['running_var'] = var
            else:
                cache['running_mean'] = momentum * cache['running_mean'] + (1 - momentum) * mu
                cache['running_var'] = momentum * cache['running_var'] + (1 - momentum) * var
    else:
        x_norm = (x_reshaped - cache['running_mean']) / cp.sqrt(cache['running_var'] + eps)
        out = gamma * x_norm + beta
    
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return out

def batch_norm_backward(dout, gamma, cache):
    """
    Perform backward pass of batch normalization.
    
    Args:
        dout: Gradient of loss w.r.t. output
        gamma: Scale parameter
        cache: Cache from forward pass
    
    Returns:
        dx: Gradient w.r.t. input
        dgamma: Gradient w.r.t. scale parameter
        dbeta: Gradient w.r.t. shift parameter
    """
    N, C, H, W = dout.shape
    dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    
    x_norm = cache['x_norm']
    mu = cache['mu']
    var = cache['var']
    x = cache['x']
    eps = 1e-5
    
    dgamma = cp.sum(dout_reshaped * x_norm, axis=0)
    dbeta = cp.sum(dout_reshaped, axis=0)
    
    dx_norm = dout_reshaped * gamma
    dvar = cp.sum(dx_norm * (x - mu) * -0.5 * (var + eps)**(-1.5), axis=0)
    dmu = cp.sum(dx_norm * -1 / cp.sqrt(var + eps), axis=0) + dvar * cp.mean(-2 * (x - mu), axis=0)
    dx = dx_norm / cp.sqrt(var + eps) + dvar * 2 * (x - mu) / (N * H * W) + dmu / (N * H * W)
    
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return dx, dgamma, dbeta

class ConvNet:
    def __init__(self, num_filters1=32, num_filters2=32, dropout_rate=0.3, num_classes=10):
        """
        Initialize the ConvNet model.
        
        Args:
            num_filters1: Number of filters in first conv layer
            num_filters2: Number of filters in second conv layer
            dropout_rate: Dropout rate for training
            num_classes: Number of output classes
        """
        self.num_filters1 = num_filters1
        self.num_filters2 = num_filters2
        self.dropout_rate = dropout_rate

        self.w1 = cp.random.randn(num_filters1, 3, 3, 3) * cp.sqrt(2.0 / (3 * 3 * 3))
        self.b1 = cp.zeros(num_filters1)
        self.gamma1 = cp.ones(num_filters1)
        self.beta1 = cp.zeros(num_filters1)
        self.bn_cache1 = {}

        self.w2 = cp.random.randn(num_filters2, num_filters1, 3, 3) * cp.sqrt(2.0 / (num_filters1 * 3 * 3))
        self.b2 = cp.zeros(num_filters2)
        self.gamma2 = cp.ones(num_filters2)
        self.beta2 = cp.zeros(num_filters2)
        self.bn_cache2 = {}

        fc_input_size = num_filters2 * 9 * 9
        self.w3 = cp.random.randn(fc_input_size, num_classes) * cp.sqrt(2.0 / fc_input_size)
        self.b3 = cp.zeros(num_classes)
    
    def forward(self, x, training=True, return_features=False):
        """
        Perform forward pass through the network.
        
        Args:
            x: Input data
            training: Whether in training mode
        
        Returns:
            y: Output probabilities
        """
        self.x1 = conv_forward(x, self.w1, self.b1)
        self.bn1 = batch_norm_forward(self.x1, self.gamma1, self.beta1, training=training, cache=self.bn_cache1)
        self.h1 = relu(self.bn1)
        self.p1, self.mask1 = maxpool_forward(self.h1, pool_size=2, stride=2)
        self.x2 = conv_forward(self.p1, self.w2, self.b2)
        self.bn2 = batch_norm_forward(self.x2, self.gamma2, self.beta2, training=training, cache=self.bn_cache2)
        self.h2 = relu(self.bn2)
        self.p2, self.mask2 = maxpool_forward(self.h2, pool_size=2, stride=2)
        self.flat = self.p2.reshape(len(x), -1)

        if training:
            self.drop_mask = cp.random.rand(*self.flat.shape) > self.dropout_rate
            self.flat = self.flat * self.drop_mask / (1 - self.dropout_rate)
        self.logits = fc_forward(self.flat, self.w3, self.b3)
        self.y = softmax(self.logits)
    
        if return_features:
            features = {
                'conv1': self.x1,
                'pool1': self.p1,
                'conv2': self.x2,
                'pool2': self.p2
            }
            return self.y, features
        
        return self.y
    
    def backward(self, x, y_true, reg):
        """
        Perform backward pass through the network.
        
        Args:
            x: Input data
            y_true: True labels (one-hot encoded)
            reg: Regularization strength
        
        Returns:
            Gradients for all parameters
        """
        m = len(x)
        dy = self.y - y_true
        dx, dw3, db3 = fc_backward(dy, self.flat, self.w3)
        
        if hasattr(self, 'drop_mask'):
            dflat = dx * self.drop_mask / (1 - self.dropout_rate)
        else:
            dflat = dx
        
        dflat = dflat.reshape(self.p2.shape)
        dp2 = maxpool_backward(dflat, self.h2, self.mask2, pool_size=2, stride=2)
        dh2 = relu_deriv(self.bn2) * dp2
        dbn2, dgamma2, dbeta2 = batch_norm_backward(dh2, self.gamma2, self.bn_cache2)
        dx2, dw2, db2 = conv_backward(dbn2, self.p1, self.w2)
        dp1 = maxpool_backward(dx2, self.h1, self.mask1, pool_size=2, stride=2)
        dh1 = relu_deriv(self.bn1) * dp1
        dbn1, dgamma1, dbeta1 = batch_norm_backward(dh1, self.gamma1, self.bn_cache1)
        _, dw1, db1 = conv_backward(dbn1, x, self.w1)
        
        dw1 += reg * self.w1
        dw2 += reg * self.w2
        dw3 += reg * self.w3
        
        return dw1, db1, dw2, db2, dw3, db3, dgamma1, dbeta1, dgamma2, dbeta2
    
    def update(self, grads, lr, max_grad_norm=5.0):
        dw1, db1, dw2, db2, dw3, db3, dgamma1, dbeta1, dgamma2, dbeta2 = grads
        for grad in [dw1, dw2, dw3, dgamma1, dbeta1, dgamma2, dbeta2]:
            norm = cp.linalg.norm(grad)
            if norm > max_grad_norm:
                grad *= max_grad_norm / norm
        self.w1 -= lr * dw1
        self.b1 -= lr * db1
        self.w2 -= lr * dw2
        self.b2 -= lr * db2
        self.w3 -= lr * dw3
        self.b3 -= lr * db3
        self.gamma1 -= lr * dgamma1
        self.beta1 -= lr * dbeta1
        self.gamma2 -= lr * dgamma2
        self.beta2 -= lr * dbeta2