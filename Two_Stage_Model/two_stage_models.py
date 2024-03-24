import torch
import numpy as np
from tqdm import tqdm

"""
Two-stage model for predicting stock direction and mangitude and whether or not to conduct a trade.

StageOneModel: Predicts the direction and magntiude of the stock price movement. The output of the model is 4 nodes.
The first three nodes represent the probability of the stock price moving up, down, or staying the same.
The fourth node represents the magnitude of the stock price movement.
If the stock direction and movement are in the same direction, the model will output will be passed to the StageTwoModel.

The input of the model is the stock price and volume data, alongside sentiment analysis and other market indicators.

The loss function for the model is a combination of the cross-entropy loss and the mean squared error loss.


StageTwoModel: Predicts whether or not to conduct a trade. The output of the model is 2 nodes.
The first node represents the probability of conducting a trade.
The second node represents the probability of not conducting a trade.

The input of the model is the stock price movement and magnitude from the StageOneModel, along with volatility and other market indicators.

The loss function for the model is the cross-entropy loss.

"""

class StageOneModel(torch.nn.Module):
    def __init__(self, input_size):
        super(StageOneModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, 4)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        

        # DIR DIR DIR MAG

        # Split the output into two parts
        direction = x[:, :3]
        magnitude = x[:, 3:]

        # Apply the sigmoid function to the direction
        direction = self.softmax(direction)

        # Concatenate the direction and magnitude
        x = torch.cat((direction, magnitude), 1)

        return x
    
'''
 ONCE A DAY <market features> -> Stage One -> Direction + Magntiude (11 AM - 4 PM movement) -> IF Direction and Mangitude AGREE -> Stage Two -> "Yes/No Buy at 11 AM, sell at 4 PM" vs "Do Nothing, Buy, Sell"
                                                                                                                                |
                                                                                                                        <other market features>

                                                                                                                        


Every X minutes <market features> -> Stage One/Stage Two -> "Do Nothing, Buy, Sell"

Enter Market: <market features> -> Model -> Long / Short / Nothing
                            |
                            V
Exit Market:  <market features> -> Model -> Sell Long / Nothing
                                            Sell Short / Nothing


Classical:      <market features> -> Optimized Algorithm -> Do Nothing, Buy, Sell

Features:
Twitter Trends?


Bubble Indicator (pos/neg)
Stumpy Indicator 
Volatility

Percentage Change Price
Volume
RSI
VWAP

----------------


Distance from MacD

Enter Market: <market features> -> Model -> Buy Long / Nothing                                 
                            |
                            V
Exit Market:  <buy time market features, current market features> -> Model -> Sell / Nothing 
'''
class StageTwoModel(torch.nn.Module):
    def __init__(self):
        super(StageTwoModel, self).__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
    
    # If we make a trade, our loss will be if we make a loss or not
    # If we don't make a trade, our loss will be                        