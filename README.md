# House Price Prediction Using Deep Learning

A Deep Learning project that predicts California house values using an Artificial Neural Network (ANN) built with TensorFlow and Keras. The project covers the complete workflow, from data preprocessing and model training to evaluation and real-time predictions through a Flask web application.

## Project Overview

House price prediction is a regression problem that involves estimating property values based on relevant housing and geographical features.

This project uses the California Housing dataset to train an ANN that learns patterns from historical housing data and predicts median house values based on user-provided inputs.

The project follows an end-to-end machine learning pipeline:

**Data Loading → Preprocessing → Model Training → Evaluation → Flask API → Web Interface**

## Features

* California Housing dataset
* Artificial Neural Network implemented using TensorFlow and Keras
* Data preprocessing and feature scaling using Scikit-learn
* Missing-value handling
* Dropout and Batch Normalization for model regularization and training stability
* Early Stopping and Learning Rate Reduction
* Regression evaluation using MAE, RMSE, and R² Score
* Flask API for real-time predictions
* Frontend interface built with HTML, CSS, and JavaScript
* Model and preprocessing artifact persistence

## Technologies Used

| Technology           | Purpose                                 |
| -------------------- | --------------------------------------- |
| Python               | Core programming language               |
| TensorFlow and Keras | Neural network development and training |
| NumPy                | Numerical computations                  |
| Pandas               | Data manipulation and analysis          |
| Scikit-learn         | Data preprocessing and model evaluation |
| Joblib               | Saving preprocessing objects            |
| Flask                | Backend API and application server      |
| HTML                 | Frontend structure                      |
| CSS                  | Frontend styling                        |
| JavaScript           | Frontend interactivity                  |

## Dataset

The project uses the California Housing dataset provided by Scikit-learn.

### Input Features

The dataset contains eight input features:

| Feature      | Description                                    |
| ------------ | ---------------------------------------------- |
| `MedInc`     | Median income of households in the block group |
| `HouseAge`   | Median house age                               |
| `AveRooms`   | Average number of rooms per household          |
| `AveBedrms`  | Average number of bedrooms per household       |
| `Population` | Block group population                         |
| `AveOccup`   | Average household occupancy                    |
| `Latitude`   | Geographic latitude                            |
| `Longitude`  | Geographic longitude                           |

### Target Variable

`MedHouseVal` represents the median house value in units of $100,000.

For example, a predicted value of `3.5` corresponds to an estimated value of $350,000.

The prediction represents the target value learned from the dataset and should not be interpreted as a guaranteed current market price.

## Model Architecture and Training

The project uses an Artificial Neural Network for a supervised regression task.

The training pipeline includes:

1. Loading the California Housing dataset.
2. Inspecting and preprocessing the input features.
3. Handling missing values, where applicable.
4. Scaling features to improve neural network training.
5. Splitting the dataset into training and testing subsets.
6. Training the ANN using TensorFlow and Keras.
7. Applying regularization and training callbacks, where configured.
8. Evaluating model performance using regression metrics.
9. Saving the trained model and preprocessing artifacts for inference.

The saved model is subsequently loaded by the Flask application to generate predictions for new input data.

## Project Structure

```text
house-pridiction/
├── app.py
├── train.py
├── index.html
├── house_price_ann.keras
├── house_price_preprocessor.joblib
└── README.md
```

| File                              | Description                                                                                               |
| --------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `train.py`                        | Loads the dataset, preprocesses the data, trains the ANN, evaluates its performance, and saves the model. |
| `app.py`                          | Runs the Flask application and handles prediction requests.                                               |
| `index.html`                      | Provides the web interface for entering housing features and viewing predictions.                         |
| `house_price_ann.keras`           | Saved TensorFlow/Keras model generated during training.                                                   |
| `house_price_preprocessor.joblib` | Saved preprocessing object used to transform input features consistently during inference.                |
| `README.md`                       | Project documentation.                                                                                    |

The trained model and preprocessing files are generated by the training script and may not be present in a fresh repository clone.

## Installation and Setup

### Prerequisites

* Python 3.10 or a compatible version supported by the installed dependencies
* pip package manager

### 1. Clone the Repository

```bash
git clone https://github.com/priyanshukumarverma091-hub/house-pridiction.git
```

Navigate to the project directory:

```bash
cd house-pridiction
```

### 2. Install Dependencies

```bash
pip install flask tensorflow numpy pandas scikit-learn joblib
```

### 3. Train the Model

Run the training script:

```bash
python train.py
```

This step preprocesses the dataset, trains the ANN, evaluates the model, and saves the trained model and preprocessing artifacts.

### 4. Start the Application

Run the Flask application:

```bash
python app.py
```

### 5. Access the Web Interface

Open the following address in your browser:

```text
http://127.0.0.1:5000
```

The application can then be used to submit housing features and obtain predictions from the trained model.

## Model Evaluation

The model is evaluated using three standard regression metrics.

### Mean Absolute Error (MAE)

Measures the average absolute difference between the actual and predicted values. Lower values indicate smaller average prediction errors.

### Root Mean Squared Error (RMSE)

Measures the square root of the average squared prediction errors. It penalizes larger errors more heavily than MAE.

### R² Score

Measures how much of the variation in the target variable is explained by the model. A higher score generally indicates a better fit, although it should be interpreted alongside other evaluation metrics.

**Evaluation results:** Actual MAE, RMSE, and R² values should be reported after evaluating the trained model on held-out test data.

## Future Improvements

* Optimize the ANN architecture and hyperparameters.
* Compare performance against traditional regression algorithms.
* Improve generalization through systematic validation and regularization.
* Add interactive data visualizations.
* Improve frontend responsiveness and usability.
* Strengthen API input validation and error handling.
* Deploy the application to a cloud platform.
* Add automated testing and continuous integration.

## Author

**Priyanshu Kumar Verma**

AI/ML Research Enthusiast | Deep Learning | Generative AI

GitHub: [priyanshukumarverma091-hub](https://github.com/priyanshukumarverma091-hub)

## License

A license can be added to the repository to specify the terms under which the project may be used, modified, and distributed.

---

If you find this project useful, consider starring the repository.
