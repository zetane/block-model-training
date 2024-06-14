# House Price Prediction using Neural Networks

This project trains and evaluates a neural network to predict house prices using the Boston Housing dataset. The script uses PyTorch for building and training the neural network and scikit-learn for data preprocessing.

## Requirements

- Python 3.x
- PyTorch
- scikit-learn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository/house-price-prediction.git
   cd house-price-prediction
   ```

2. Install the required packages:
   ```bash
   pip install torch scikit-learn
   ```

## Usage

### Function: `compute`

The `compute` function trains and evaluates a neural network model for house price prediction.

#### Parameters

- `epoch` (int): The number of training epochs.

#### Returns

- `result` (dict): A dictionary containing the average loss on the validation set.

### Example

```python
from your_module import compute

result = compute(epoch=100)
print(result)
```

### Function: `test`

The `test` function runs a simple test to ensure that the `compute` function works correctly.

```python
from your_module import test

test()
```

## Code Explanation

### Data Loading and Preprocessing

1. **Loading the Dataset**: The Boston Housing dataset is loaded using scikit-learn's `load_boston` function.
2. **Preprocessing**: The features are standardized using `StandardScaler`. The dataset is split into training and testing sets using `train_test_split`.
3. **Creating Datasets and DataLoaders**: The data is converted to PyTorch tensors and wrapped in `DataLoader` objects for easy batch processing.

### Neural Network Definition

A simple feedforward neural network with one hidden layer is defined using PyTorch's `nn.Module`. The network consists of:
- An input layer
- A hidden layer with ReLU activation
- An output layer

### Training

The network is trained using mean squared error (MSE) loss and the Adam optimizer. The training loop iterates through the data for the specified number of epochs, updating the model parameters to minimize the loss.

### Evaluation

The trained model is evaluated on the test set, and the average loss is computed and returned.
