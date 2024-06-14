def compute(epoch):
    """Trains and evaluates a neural network for house prediction.

    Inputs:
        epoch (int): The number of training epochs.

    Outputs:
        result (dict): A dictionary containing the validation loss.

    Requirements:
        PyTorch, scikit-learn
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset

    # 1. Setup and Data Loading
    # Load the Boston housing dataset
    data = load_boston()

    # Preprocessing
    X = data.data
    y = data.target

    # Splitting data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardizing features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Converting data into PyTorch tensors and creating datasets
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 2. Define the Neural Network
    class Net(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            return out

    input_size = X_train.shape[1]
    hidden_size = 50
    output_size = 1
    model = Net(input_size, hidden_size, output_size)

    # 3. Training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = epoch
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # 4. Evaluation
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for inputs, labels in test_loader:
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item()

    average_loss = total_loss / len(test_loader)
    print(f"Average loss on test data: {average_loss:.4f}")

    return {"avg_loss": str(average_loss)}


def test():
    """Test the compute function."""
    print("Running test")
