# Traffic Prediction in Telecom Systems Using Deep Learning and XGBoost

## Dataset

The dataset used in this project can be accessed from the following link:
https://www.kaggle.com/datasets/ocanaydin/italian-telecom-data-2013-1week 

## I. Introduction

**Project Objective:**  
The aim of this project is to predict traffic in telecom systems using machine learning and deep learning models.

**Models Used:**  
In this project, we utilized XGBoost as the machine learning model, and ANN (Artificial Neural Network), CNN (Convolutional Neural Network), and RNN (Recurrent Neural Network) as the deep learning models.

**Dataset Description:**  
The dataset used in this project includes information about SMS, calls, and internet usage in telecommunication systems. It covers daily activities from November 4 to November 10, 2013. Each row in the dataset represents the daily activities of a specific user.

**Variables in the Dataset:**

- `smsin`: Number of received SMS
- `smsout`: Number of sent SMS
- `callin`: Number of incoming calls
- `callout`: Number of outgoing calls
- `internet`: Amount of internet data used (in MB)

The dataset was also used to calculate the total activity for each user, labeled as `total_activity`, which is the sum of all SMS, call, and internet usage.

Users were categorized into different activity levels based on their total activity values:
- 1: Total activity between 0-20
- 2: Total activity between 20-40
- 3: Total activity between 40-60
- 4: Total activity between 60-80
- 5: Total activity between 80-100
- 6: Total activity above 100

This dataset was used to understand and predict users' activities on telecommunication systems. Our goal was to classify user activities using various machine learning and deep learning models trained with this dataset.

## II. Data Preparation

**Data Merging and Cleaning:**  
The dataset consists of seven parts, each corresponding to one day. These datasets were initially stored separately as DataFrames. We combined them into a single DataFrame using the `concat` function from the Pandas library. After merging, we proceeded with data cleaning by removing empty values using the `dropna` function.

**Defining Dependent and Independent Variables:**  
The independent variables are `smsin`, `smsout`, `callin`, `callout`, and `internet`, collectively referred to as `X`. The dependent variable is `activity_number`, denoted as `y`. We then split these `X` and `y` variables into `X_train`, `X_test`, `y_train`, and `y_test`. Standardization was applied to the `X_train` and `X_test` variables afterward.

## III. Model Information

- **XGBoost (Extreme Gradient Boosting):** An optimized version of the gradient boosting algorithm, XGBoost is a powerful machine learning model capable of making fast and high-performance predictions. It is a tree-based ensemble model often used in competitive data science. Key features include parallel computation, sparsity awareness, and parameters that control overfitting.

- **ANN (Artificial Neural Network):** Inspired by biological neural networks, ANNs are composed of artificial neurons arranged in layers. Input data passes through hidden layers to make a prediction at the output layer. ANNs are known for their ability to learn nonlinear relationships and perform well on complex data structures.

- **CNN (Convolutional Neural Network):** CNNs are neural networks particularly effective for image data. They use convolutional layers to learn spatial relationships and features in images by applying filters to input data. Pooling layers reduce the dimensionality and computational load. CNNs are widely used in image classification, object recognition, and image processing.

- **RNN (Recurrent Neural Network):** RNNs are neural networks designed to process sequential and time-series data. They use feedback mechanisms to capture dependencies over time steps, making them useful for tasks like language modeling, speech recognition, and time-series forecasting. Advanced RNN types like LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) were developed to address challenges in learning long-term dependencies.

## IV. Model Performance Comparison

**Training Times:**
- XGBoost training time: 49.3 seconds
- ANN training time: 19.7 seconds
- CNN training time: 35.2 seconds
- RNN training time: 206.3 seconds

**Number of Parameters:**
- ANN parameter count: 3,094
- CNN parameter count: 4,422
- RNN parameter count: 8,214

**Total FLOP Count:**
- XGBoost FLOP count: 12,700
- ANN FLOP count: 3,094
- CNN FLOP count: 1,878
- RNN FLOP count: 3,094

## V. Results and Discussion

**Overall Model Performance:**  
The accuracy values of the models are very close, all around 0.99. However, training times, parameter counts, and total FLOP values differ among the models. ANN performed the best in terms of training time, while RNN took significantly longer to train. Thus, except for RNN, the other models can be considered fast. The training times of ANN and the other models are about one-fifth of that of RNN. In terms of parameter count, ANN has the fewest, which offers advantages like faster training and prediction. RNN has the highest parameter count, which can lead to both advantages and disadvantages, such as higher capacity but slower speed. Considering the FLOP counts, CNN is the most advantageous model, requiring fewer computations, thus being efficient in terms of CPU usage. On the other hand, XGBoost has a very high FLOP count, putting a heavy load on the CPU.

**Best Model Selection and Reasons:**  
Based on the results, ANN appears to be the best model. It has a low training time, making it fast, and fewer parameters, contributing to its efficiency. Additionally, its FLOP count is not excessively high, meaning it does not overburden the CPU. Considering all these factors, ANN is both fast and CPU-efficient.
