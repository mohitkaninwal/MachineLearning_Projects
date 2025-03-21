# ANN MNIST Classification

## Project Overview
This project implements an **Artificial Neural Network (ANN)** to classify handwritten digits from the **MNIST dataset**. The MNIST dataset consists of 60,000 training images and 10,000 test images of handwritten digits (0-9). The goal of this project is to train an ANN model to accurately classify the digits.
![ANN_MNIST](https://github.com/user-attachments/assets/101d4a35-a2e6-4a0b-8ad3-e06a89ace016)


## Dataset
- **MNIST Handwritten Digits Dataset**
- Contains **28x28 grayscale images** of digits from 0 to 9
- 60,000 training images and 10,000 test images
- Available in TensorFlow/Keras and PyTorch datasets

## Model Architecture
- **Input Layer:** 784 neurons (flattened 28x28 images)
- **Hidden Layers:**
  - Dense Layer (128 neurons, ReLU activation)
  - Dense Layer (64 neurons, ReLU activation)
- **Output Layer:** 10 neurons (softmax activation for classification)

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib (for visualization)
- Google Colab (for training)

## Steps Involved
1. **Data Preprocessing:**
   - Load the dataset
   - Normalize pixel values to the range [0,1]
   - Reshape images for ANN input
2. **Building the ANN Model:**
   - Define the model architecture
   - Compile the model (Optimizer: Adam, Loss: Categorical Crossentropy)
3. **Training the Model:**
   - Train on the MNIST dataset
   - Monitor accuracy and loss
4. **Evaluation and Testing:**
   - Evaluate model performance on test data
   - Visualize predictions
5. **Predictions on New Data:**
   - Test with user-provided images

## Results
- Achieved **high accuracy (>97.8%)** on test data
- Successfully classified handwritten digits with good generalization

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/mohitkaninwal/MachineLearning_Projects.git
   ```
2. Navigate to the project directory:
   ```bash
   cd MachineLearning_Projects
   ```
3. Open the Jupyter Notebook or Google Colab and run the code.

## Future Improvements
- Experiment with **CNN models** for better accuracy
- Implement **data augmentation** to improve generalization
- Deploy the model using **Flask or FastAPI**



