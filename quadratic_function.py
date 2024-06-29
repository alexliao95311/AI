import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Generate dataset with higher density around -0.5 to 0.5
x1 = torch.arange(-100, -0.5, 0.1, dtype=torch.float32).view(-1, 1)
x2 = torch.arange(-0.5, 0.5, 0.01, dtype=torch.float32).view(-1, 1)
x3 = torch.arange(0.5, 100, 0.1, dtype=torch.float32).view(-1, 1)
x = torch.cat((x1, x2, x3), dim=0)
y = x**2

# Normalize the data
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_normalized = torch.tensor(scaler_x.fit_transform(x), dtype=torch.float32)
y_normalized = torch.tensor(scaler_y.fit_transform(y), dtype=torch.float32)

# Define a simpler model with ReLU activation
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(1, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

model = SimpleNN()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with early stopping
target_loss = 0.0000001
train_losses = []
max_epochs = 10000  # Max number of epochs to prevent infinite loop

for epoch in range(max_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_normalized)
    loss = criterion(outputs, y_normalized)
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    if loss.item() < target_loss:
        print(f'Stopping early at epoch {epoch + 1} with loss {loss.item():.6f}')
        break
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}], Loss: {loss.item():.6f}')

print('Finished Training')

# Make predictions
model.eval()
with torch.no_grad():
    predictions_normalized = model(x_normalized)

# Inverse transform predictions to original scale
predictions = scaler_y.inverse_transform(predictions_normalized.numpy())
y_actual = y.numpy()
x_actual = x.numpy()

# Filter data to the desired x range from -10 to 10
filter_indices = (x_actual >= -10) & (x_actual <= 10)
x_filtered = x_actual[filter_indices]
y_filtered = y_actual[filter_indices]
predictions_filtered = predictions[filter_indices]

# Calculate Mean Relative Percentage Error (MRPE)
def mrpe(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Exclude points where y_actual is zero to avoid division by zero
non_zero_indices = y_filtered != 0
y_actual_non_zero = y_filtered[non_zero_indices]
predictions_non_zero = predictions_filtered[non_zero_indices]

mrpe_value = mrpe(y_actual_non_zero, predictions_non_zero)
print(f'MRPE: {mrpe_value:.2f}%')

# Write x, y, and predicted y to file
with open('relative_percentage_errors.txt', 'w') as f:
    f.write('x, y, predicted_y\n')
    for x_val, y_val, pred_val in zip(x_filtered, y_filtered, predictions_filtered):
        f.write(f'{x_val:.6f}, {y_val:.6f}, {pred_val:.6f}\n')

# Plot the true function vs the model's predictions
plt.figure(figsize=(10, 6))
plt.plot(x_filtered, y_filtered, label='True function (y = x^2)', color='blue')
plt.plot(x_filtered, predictions_filtered, label='Model predictions', color='red', linestyle='dashed')
plt.xlabel('x values')
plt.ylabel('y values')
plt.title(f'True function vs Model predictions (MRPE: {mrpe_value:.2f}%)')
plt.xticks(np.arange(-10, 11, step=1))
plt.yticks(np.arange(0, 101, step=10))
plt.legend()
plt.show()
