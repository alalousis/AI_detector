# -----------------------------------------------------------------------------
#                          MODEL EVALUATION PROCESS:
# -----------------------------------------------------------------------------
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import datetime as dt

# This function returns the available devices cuda gpu, apple gpu or simple
# cpu that will be used to train the model.
def get_execution_device():
    # Set the existence status of a mps GPU.
    if hasattr(torch.backends,"mps"):
        is_mps = torch.backends.mps.is_available()
    else:
        is_mps = False
    # Set the existence status of a cuda GPU.
    is_cuda = torch.cuda.is_available()
    # Check the existence status of a mps GPU to be used during training.
    if is_mps:
        device = torch.device("mps")
        print("MPS GPU is available!")
        print(70*"=")
    # Check the existence of a cuda GPU to be used during training.
    elif is_cuda:
        device = torch.device("cuda")
        print("CUDA GPU is available!")
        print(70*"=")
    # Otherwise, a CPU device will be used instead.
    else:
        device = torch.device("cpu")
        print("GPU is not available, CPU will be used instead!")
        print(70*"=")
    return device


# -----------------------------------------------------------------------------
# Define the Sentiment Classifier Class.
# -----------------------------------------------------------------------------
class SentimentClassifier(nn.Module):

    # Class constructor.
    def __init__(self, bert_model, n_classes, dropout_percent=0.3, freeze_bert=False):
        # Call the super class constructor.
        super(SentimentClassifier, self).__init__()
        self.bert = bert_model
        # Freeze the parameters of the BERT model in the case the corresponding
        # input argument is True.
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.drop = nn.Dropout(p=dropout_percent)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    # Define the function that describes the forward pass of information within
    # the network. Mind that the network module is being fed with the input ids
    # and the attention mask provided by the BERT tokenizer for each text.
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        # When return_dict=False, the BERT model returns a tuple of two elements:
        # [1]: The hidden state of all layers of the neural model.
        # [2]: The pooled output, which is the representation of the input
        #      sequence after being processed by the model.
        # By setting return_dict=False, we are explicitly instructing the BERT
        # model to return a tuple of outputs instead of a dictionary. This is
        # often done for compatibility with older versions of the Hugging Face
        # Transformers library or for custom model implementations that expect
        # tuples as outputs. If return_dict=True, the BERT model would return a
        # dictionary containing various outputs, such as the hidden states,
        # pooled output, and other intermediate outputs.
        output = self.drop(pooled_output)
        return self.out(output)


def evaluate_model(model, data_loader, device, return_probabilities=False):
    # Set the model evaluation environment.
    model.eval()
    # Initialize variables counting the total and correctly classified text
    # instances.
    total, correct = 0, 0
    # Initialize the list containing the output probabilities for each text
    # instance.
    all_probabilities = []

    # Indicate that no gradient-based updating of the model weight-vector will
    # be performed during this process.
    with torch.no_grad():
        for batch in tqdm(data_loader):
            # Acquire the information that the bert tokenizer associated with
            # each textual input.
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            # Acquire the network output for each instance in the batch.
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Convert the network primitive output to probabilities.
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            # Get the predicted class index by determining which component of
            # the probability vector contains the maximum value.
            _, predicted = torch.max(probabilities, 1)
            # If the corresponding input argument indicates that probability
            # vectors should also be returned, then perform the necessary
            # computation.
            if return_probabilities:
                all_probabilities.extend(probabilities.cpu().numpy())
            # Accumulate the total number of text instances that have been
            # classified so far.
            total += labels.size(0)
            # Accumulate the total number of text instances that have been
            # correctly classified.
            correct += (predicted == labels).sum().item()

    # Compute the total accuracy of the model on the subset of text instances
    # stored in the data loader.
    accuracy = correct / total

    return (accuracy, np.array(all_probabilities)) if return_probabilities else accuracy


# -----------------------------------------------------------------------------
#                          MODEL TRAINING PROCESS:
# -----------------------------------------------------------------------------

def train_model(model, train_loader, test_loader, optimizer, loss_fn, epochs, device,
                checkpoint_path=None, batch_save_period=None):
    # Load the state variables from the last training session.
    global datetime
    checkpoint = torch.load(checkpoint_path)
    # Load the last state of the neural model.
    model.load_state_dict(checkpoint["model_state_dict"])
    # Load the last state of the optimizer.
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # Set the starting epoch of the training process to be the last
    # training epoch of the previous session.
    start_epoch = checkpoint["epoch"]
    # Load the last batch processed during the previous training session.
    batch_count = checkpoint["batch_count"]
    # Load the train accuracy achieved by the model so far.
    train_accuracy = checkpoint['train_accuracy']
    # Load the test accuracy achieved by the model so far.
    test_accuracy = checkpoint["test_accuracy"]

    # Actual Model Training Process

    # Loop through the remaining training epochs.
    for epoch in range(start_epoch, epochs):
        # Set the model training environment.
        model.train()
        # Loop through the various batches.
        for batch_idx, batch in enumerate(tqdm(train_loader,
                                               desc=f"Epoch: {epoch + 1} / {epochs} train_acc: {train_accuracy} test_acc: {test_accuracy}")):
            # Skip batches until the last processed batch is reached.
            if batch_idx < batch_count:
                continue
            # Clear gradients.
            optimizer.zero_grad()
            # Load the necessary information from the current batch.
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            # Compute the output of the model and the associated loss by
            # performing the forward pass of information.
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            # Optimize model parameters by performing the backward pass of
            # information.
            loss.backward()
            optimizer.step()
            # Increase the number of batches processed so far.
            batch_count += 1
            # Save checkpoint if specified and if the currently processed batch
            # is an integer multiple of the save_every_batches input arguments.
            # Thus, a training checkpoint will be saved after every given amount
            # of batches has been processed.
            if batch_save_period is not None and batch_count % batch_save_period == 0:
                torch.save({
                    'epoch': epoch,
                    'batch_count': batch_count,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy
                }, checkpoint_path)

        # After the termimation of each training epoch the batch_count variable
        # must be reset to zero so that the next training epoch does not skip
        # all the batches that have been completed by the previous session.
        batch_count = 0
        # Evaluate the train accuracy of the model after each epoch.
        train_accuracy = evaluate_model(model, train_loader, device)
        # Evaluate the model on the test set after each epoch.
        test_accuracy = evaluate_model(model, test_loader, device)
        # Save final checkpoint after each training epoch has been completed.
        torch.save({
            'epoch': epoch,
            'batch_count': batch_count,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
        }, checkpoint_path)

        datetime = dt.datetime.now()
        history_file_path = f"/Users/alalousis/PycharmProjects/AI_detector/results/checkpoint_{datetime}.txt"

        with open(history_file_path, 'w') as file:
            file.write(f"epoch: {epoch+1}\n")
            file.write(f"batch_count: {batch_count}\n")
            file.write(f"train_accuracy: {train_accuracy}\n")
            file.write(f"test_accuracy: {test_accuracy}\n")


    # Report the termination of the training process.
    print("Training process completed")