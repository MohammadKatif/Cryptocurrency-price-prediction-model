# Cryptocurrency Price Prediction Model
This is a Deep learning project made using Pytorch and some additional libraries which basically functions to accurately forceast future price of the Cryptocurrency **"Ethereum" in USD**  
Since the data used in this project will be timeseries data so I will be using Long Short Term Memory(LSTM) model throught this project.
For this project I learned about Deep learning from the [Official 60 min blitz tutorial](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) and the [Official Pytorch documentation](https://pytorch.org/docs/stable/index.html) made by the Pytorch developers themselves and I also took help from [Peter](https://github.com/VaguelySerious/) who is a software engineer and has already studied about this field, Thanks alot Peter!
## Background about Deep learning:
* Neural Network is just a collection of nested functions.  
* These functions are defined by **Parameters(Weight and Bias)**. And these parameters themselves are stored inside Pytorch tensors.  
* Pytorch tensors are a specialized datastructure similar to numpy arrays. The inputs and outputs of a NN model and the parameters of the functions are **encoded in tensors** in Pytorch. (Tensors can also run on GPU)
## How a Neural Network is Trained?
### Forward Propagation:
* In Forward propagation the model **makes its best guess** about the correct target output.
* It does this by running the input data through each of these functions with their corresponding parameters to the best guess about the output.  
### Back Propagation:
* In Backward propagation the model **adjusts the parameters** of its functions according to the error/ loss it made in its guess.
* It does this by **moving backwards** from the output of the model, collecting the **derivative/ gradient** of the function which was responsible for the error/ loss which is also known as the **'loss function'**, and then optimizing the parameters of the loss function using **gradient descent**.
## Crypto Model 1:
This is the very first code that I had written for this project, where I discovered the Open, High, Low, Close or also known as **OHLC** dataset of the cryptocurrency "Etherum". And I also explored two libraries for plotting the financial data, from which the first one is [Mplfinance](https://coderzcolumn.com/tutorials/data-science/candlestick-chart-in-python-mplfinance-plotly-bokeh#) which offers multiple types and styles of graph to plot financial data, from which my favourite type is 'candle' and style is 'tradingview' but it lacks customisation of the graph which makes it a very limited library for plotting the graphs. And the second library that I explored is [Matplotlib](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html) which is a very big library and offers a quite large range of customization for the graphs which makes it very useful for my project. Nevertheless, I will be using both of these libraries throughout the code of this project.
## Crypto Model 2:
In this code I used [Yahoo Finance API](https://pypi.org/project/yfinance/) to fetch the historical OHLC dataset of Ethereum in USD. Furthermore, I also discovered a new library known as ["Technical Analysis Library"](https://ta-lib.github.io/ta-lib-python/) or ["TA-Lib"](https://pypi.org/project/TA-Lib/), which was very difficult to download untill I watched this [youtube video](https://www.youtube.com/watch?v=30BaSfz0FGE&t=285s). TA-Lib is a very interesting library as it offers **candlestick pattern recognition** and **more than 150 indicators**. The benefit of this library is that it does all the complicated calculation to for candlestick patterns and indicators so you don't have to! In this code, I planned on using these candlestick patterns and indicators as features for the model but later on I understood the **Beauty of Deep learning** which makes this library almost useless.
## Crypto Model 3:
In this code I made my first LSTM model for which I took reference from [this article](https://cnvrg.io/pytorch-lstm/). After writing this code for a moment I thought that I have successfully completed the project, but to my dismay it was not the case here. 
### Problem:
This code lacked the "Concept of Windowing" which makes it totally useless for my project!
### The Concept of Windowing:
The model needs to look at N days of data(which is the window) and then predict N+n_ahead th day of data. The point is that, the model will predict the Close price for "n_ahead" days by looking at the "past N days" of OHLC data.
## Crypto Model 5:
For this article I took reference from two articles, the first one being the [previous article]() and the second one being this [new article]() which was suggested to me by Peter. This new article is one of the best reference for my project so far, because it has the "Concept of Windowing" in it! But the model used in model is different from the model that should be used in this project. So I used the model from the previous article's code and the windowing code from this new article.
### Problem: 
The model I am using  in this code requires the input and output both of the same size, but the input I am providing to this model has the size of 6 (Open, High, Low, Close, AdjClose, Volume) and the output has the size of 1 (Close price). That's why the Mean Sqquared Error(MSE) Loss seems to be stuck and is not reducing below 1500 in the training loop.
## Crypto Model 6:
This code is written by Peter, where he identified the issue from previous code and changed the input size to 1(Close price) so that It will match with the output size of 1 (Also Close price). Due to this the MSE Loss of the model successfully reduced from 1600 to somewhere near 100. Also in this code the model is shifted to CUDA enabled GPU because it will take a long time to train in CPU, and because my laptop doesn't have CUDA enabled GPU so I used [Google Colab](https://colab.google/) because it provides success to powerful computing resources such as CUDA enabled GPU.
### Problem:
The problem is that the input feature is only the Close price but we need the Open, High, Low and Close all as our input features which is not possible because the input size and ouput size should be same for this model.
## Crypto Model 7:
This code is also written by Peter to solve the problem that arised in his previous code. Here he changed the input features size to be 4 (Open, High, Low, Close) and the ouput prediction target size to also be 4 (Open, High, Low, Close) after this he basically ignored the Open, High, Low that were in the output and only took the 'Close price' from that predicted output and compared it with the 'Actual Close Price' in the graphing code at the end. Moreover, here he also used a Mean of Mean Squared Error or MMSE to calculate Loss for the backpropagation instead of using the normal MSE one.
### Beauty of Deep learning:
Although its not present in this code but, when I added all the identified candlestick patterns as features into this code (using the TA-Lib library) then I noticed that the graph didn't changed at all! Why is that so? because **candlestick patterns is just a thing that humans have made so that they(humans) can identify the patterns** in the financial data to predict future price, but since this is a Network so it does all the complicated calculation that the library does and successfully identifies these patterns in the data by itself. Therefore, there is no need to give candlestick patterns as extras feature to the model!
### Problem:
If I want to add more features then I would also need to make them the prediction target too, which might result in significant reduction in the accuracy of the model when using more features. (Although I haven't acutally tested whether this would affect the accuracy of the model or not)
## Crypto Model 8:
This is just a slightly **improved version of the previous code** where Peter increased the layers of the model to increase accuracy and added two more graphs into the code to get better understanding of the model.   (Once again, thanks for all the help Peter!)
### Problem:
The problem is still the same as the previous code.
## Crypto Model 9:
In this code I will be doing feature engineering to add several new features in the model and track its affect on the model performance(Error/ Loss). The purpose is to identify the features that will improve the accuracy of the model and identify the ones that will not improve the accuracy of the model.
### Seasonality:
#### Take#1: Without any additional columns for seasonality
##### Result: MAE= 20
#### Take#2: With 1 additional column: dayofweek
##### Result: MAE= 53
#### Take#3: With 1 additional column: weekofmonth
##### Result: MAE= 96
#### Take#4: With 1 additional column: monthofyear
##### Result: MAE= 73
#### Take#2: With 3 additional columns: dayofweek, weekofmonth, monthofyear
##### Result: MAE= 99
#### Outcome: For some reason seasonality is not a factor that affects the closing price of ethereum.



