import torch
from RNN.Classification.utils import line_to_tensor, load_data, N_LETTERS
from rnn import RNN, category_from_output

# Load the dataset and categories
category_lines, all_categories = load_data()

# Define hidden layer size and load the trained RNN model
n_hidden = 128
rnn = RNN(N_LETTERS, n_hidden, len(all_categories))
rnn.load_state_dict(torch.load("rnn_model.pth"))  # Load trained weights
rnn.eval()  # Set the model to evaluation mode 

# Override the original category_from_output function
# Now returns the top `k` predictions instead of just one
def category_from_output(output, topk=3):
    topv, topi = output.topk(topk, 1, True)  # Get top `k` values and indices
    predictions = []
    for i in range(topk):
        value = topv[0][i].item()  # Confidence score
        category_idx = topi[0][i].item()  # Predicted index
        predictions.append((all_categories[category_idx], value))  # (label, score)
    return predictions

# Predict the most probable categories for a given input name
def predict(input_line):
    print(f"\n> {input_line}")
    with torch.no_grad():  # Disable gradient tracking for inference
        line_tensor = line_to_tensor(input_line)
        hidden = rnn.init_hidden()

        # Forward pass through each character of the input line
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        # Get top predictions
        predictions = category_from_output(output)

        # Print each prediction with its score
        for category, score in predictions:
            print(f"{category}: {score:.4f}")

# Interactive loop to enter names and get predictions
print("Write names to predict their nationality.")
print("To STOP, enter : quit")
while True:
    sentence = input("Input: ")
    if sentence.lower() == "quit":
        break
    predict(sentence)


