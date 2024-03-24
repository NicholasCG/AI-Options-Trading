import torch
from two_stage_train import StageOneModel, StageTwoModel
import os
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


"""
StageOneModel: Predicts the direction and magntiude of the stock price movement. The output of the model is 4 nodes.
The first three nodes represent the probability of the stock price moving up, down, or staying the same.
The fourth node represents the magnitude of the stock price movement.

The input of the model is the stock price and volume data, alongside sentiment analysis and other market indicators.

The loss function for the model is a combination of the cross-entropy loss and the mean squared error loss.

The training process is different from the usual training process.
The data is time-series data, and the model will first be trained on an initial window of the data.
The training set will then be shifted by a certain number of days, and the model will be trained again, but with fewer epochs and a smaller learning rate.
This process will be repeated until the end of the data is reached.
The thought process behind this is that the model will be able to adapt to the changing market conditions, with the initial training providing a good starting point.
"""

def convert_to_windows(data, window_size):
    """
    Conver the data into windows.
    """
    
    windows = []
    for i in range(len(data) - window_size):
        window = data[i:i+window_size]
        windows.append(window)
    
    return windows

def criterion_stage_one_model(prediction, target):
    """
    The loss function for the StageOneModel.
    """
    # Split the output into two parts
    direction = prediction[:, :3]
    magnitude = prediction[:, 3:]

    # Split the target into two parts
    target_direction = target[:, :3]
    target_magnitude = target[:, 3:]

    # Calculate the loss
    direction_loss = torch.nn.CrossEntropyLoss()(direction, torch.argmax(target_direction, dim=1))
    magnitude_loss = torch.nn.MSELoss()(magnitude, target_magnitude)

    loss = direction_loss + magnitude_loss

    # note to self: punish the model more if it the direction and magnitude are not in the same direction more frequently

    return loss

def run_main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"Using {device} device")

    # First, we will load the data
    df = pd.read_csv(args.data_path)
    df["Date"] = pd.to_datetime(df["Date"])

    # Split the data to pre April 2022 and post April 2022
    train_df = df[df["Date"] < pd.to_datetime("2022-04-01")]
    test_df = df[df["Date"] >= pd.to_datetime("2022-04-01")]

    # Convert to numpy arrays
    train_df = train_df.to_numpy()
    test_df = test_df.to_numpy()

    # Convert the data into windows
    train_windows = convert_to_windows(train_df, args.window_size)

    # Create the model
    model = StageOneModel(input_size=args.window_size).to(device)

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Create the loss function
    criterion = criterion_stage_one_model

    # Create the data loader for the initial training (window 0)
    train_loader = torch.utils.data.DataLoader(train_windows[0], batch_size=args.batch_size, shuffle=True)

    # Train the model
    model.train()

    for epoch in range(args.init_epochs):
        for batch in train_loader:
            data, target = batch[:, :-1], batch[:, -1]

            # Move the data and target to the device
            data, target = data.to(device), target.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            loss = criterion(output, target)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            print(f"Epoch [{epoch}/{args.epochs}], Loss: {loss.item()}")



    """
    2016 - 2023
    All:
    ####################################################

    Rolling window:
    ################
     ################
      ################
       ################
        ################
    
    ################
        ################
            ################
                ################
                    ################
    

    ################
                    ################
                                    ################
                                                    ################
                                                                    ################


    """



    # Now for the other windows, we will train the model again, but with fewer epochs and a smaller learning rate
    for i in range(1, len(train_windows)):
        # Create the data loader
        train_loader = torch.utils.data.DataLoader(train_windows[i], batch_size=args.batch_size, shuffle=True)

        # Create the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate / 10)

        # Train the model
        for epoch in range(args.tune_epochs):
            for batch in train_loader:
                data, target = batch[:, :-1], batch[:, -1]

                # Move the data and target to the device
                data, target = data.to(device), target.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                output = model(data)

                loss = criterion(output, target)

                # Backward pass
                loss.backward()

                # Update the weights
                optimizer.step()

                print(f"Epoch [{epoch}/{args.epochs}], Loss: {loss.item()}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Two-stage model for predicting stock direction and mangitude and whether or not to conduct a trade.")

    parser.add_argument("--data_path", type=str, help="Path to the data")
    parser.add_argument("--window_size", type=int, help="Window size for the data", default=200)

    parser.add_argument("--learning_rate", type=float, help="Learning rate for the first stage model", default=0.001)
    parser.add_argument("--init_epochs", type=int, help="Number of epochs for the initial training", default=10)
    parser.add_argument("--tune_epochs", type=int, help="Number of epochs for the subsequent trainings", default=5)

    # parser.add_argument("--learning_rate_2", type=float, help="Learning rate for the second stage model", default=0.001)
    # parser.add_argument("--epochs_2", type=int, help="Number of epochs for the second stage model", default=10)

    parser.add_argument("--batch_size", type=int, help="Batch size for the model", default=32)

    parser.add_argument("--log_dir", type=str, help="Path to the log directory")

    args = parser.parse_args()

    run_main(args)
