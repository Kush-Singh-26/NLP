import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import ALL_LETTERS, N_LETTERS
from utils import load_data, line_to_tensor, random_training_example

# Main RNN implementation Class 
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        # Linear layer to compute hidden state: input + hidden -> hidden
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)

        # Linear layer to compute output: input + hidden -> output
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, input_tensor, hidden_tensor):
        # Concatenate input and hidden state
        combined = torch.cat((input_tensor, hidden_tensor), 1)

        # Compute next hidden state
        hidden = torch.tanh(self.i2h(combined))

        # Compute output
        output = self.i2o(combined)
        return output, hidden

    def init_hidden(self):
        # Initialize the hidden state with zeros
        return torch.zeros(1, self.hidden_size)


# Given output tensor, return the predicted category 
def category_from_output(output, all_categories):
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]


# One training step
def train(rnn, line_tensor, category_tensor, criterion, optimizer):
    hidden = rnn.init_hidden()

    # Forward pass through each character in the name
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    # Compute the loss using the final output
    loss = criterion(output, category_tensor)

    # Backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()


# Predict the category for an input name
def predict(rnn, input_line, all_categories):
    print(f"\n> {input_line}")
    with torch.no_grad():
        line_tensor = line_to_tensor(input_line)
        hidden = rnn.init_hidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        guess = category_from_output(output, all_categories)
        print(guess)


# Main execution block
if __name__ == "__main__":
    # Load name data grouped by category 
    category_lines, all_categories = load_data()
    n_categories = len(all_categories)

    # Create RNN instance
    n_hidden = 128
    rnn = RNN(N_LETTERS, n_hidden, n_categories)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.AdamW(rnn.parameters(), lr=learning_rate)

    current_loss = 0
    all_losses = []               # For plotting the training loss
    plot_steps, print_steps = 1000, 5000
    n_iters = 100000              # Total training iterations

    print("Training starts ")
    for i in range(n_iters):
        # Get a random training example
        category, line, category_tensor, line_tensor = random_training_example(category_lines, all_categories)

        # Train the model and accumulate loss
        output, loss = train(rnn, line_tensor, category_tensor, criterion, optimizer)
        current_loss += loss

        # Track loss for plotting
        if (i + 1) % plot_steps == 0:
            all_losses.append(current_loss / plot_steps)
            current_loss = 0

        # Print predictions and accuracy occasionally
        if (i + 1) % print_steps == 0:
            guess = category_from_output(output, all_categories)
            correct = "CORRECT" if guess == category else f"WRONG ({category})"
            print(f"{i+1} {(i+1)/n_iters*100:.1f}% {loss:.4f} {line} / {guess} {correct}")

    # Save the trained model
    torch.save(rnn.state_dict(), "rnn_model.pth")

    # Plot the training loss
    plt.figure()
    plt.plot(all_losses)
    plt.show()

    print("Write names to predict their nationality.")
    print("To STOP, enter : quit")
    # Interactive prediction loop
    while True:
        sentence = input("Input: ")
        if sentence.lower() == "quit":
            break
        predict(rnn, sentence, all_categories)
