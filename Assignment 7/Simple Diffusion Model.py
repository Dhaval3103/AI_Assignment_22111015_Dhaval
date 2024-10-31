import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Create a sine wave signal
def create_sine_wave(frequency=1, duration=1, sampling_rate=100):
    t = np.linspace(0, duration, duration * sampling_rate)
    signal = np.sin(2 * np.pi * frequency * t)
    return t, signal

# Define the diffusion model
class SimpleDiffusionModel(nn.Module):
    def __init__(self):
        super(SimpleDiffusionModel, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

# Generate a sine wave and add noise
def add_noise(signal, noise_level=0.1):
    noise = np.random.normal(0, noise_level, signal.shape)
    return signal + noise

# Main function
if __name__ == "__main__":
    # Parameters
    frequency = 1
    duration = 2
    sampling_rate = 100
    noise_level = 0.2

    # Create sine wave and add noise
    t, original_signal = create_sine_wave(frequency, duration, sampling_rate)
    noisy_signal = add_noise(original_signal, noise_level)

    # Convert to PyTorch tensors
    t_tensor = torch.tensor(t, dtype=torch.float32).view(-1, 1)
    noisy_tensor = torch.tensor(noisy_signal, dtype=torch.float32).view(-1, 1)

    # Initialize model and optimizer
    model = SimpleDiffusionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        output = model(noisy_tensor)
        loss = criterion(output, t_tensor)  # Using time as target for simplicity
        loss.backward()
        optimizer.step()

    # Denoising the signal
    model.eval()
    denoised_signal = model(noisy_tensor).detach().numpy()

    # Plotting results
    plt.figure(figsize=(12, 6))
    plt.plot(t, original_signal, label='Original Signal', color='green')
    plt.scatter(t, noisy_signal, label='Noisy Signal', color='red', alpha=0.5)
    plt.plot(t, denoised_signal, label='Denoised Signal', color='blue')
    plt.legend()
    plt.title('1D Diffusion Model: Denoising a Sine Wave')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

