# Sports Image Classification with Deep Learning

## Project Overview

This project aims to develop a robust deep learning model capable of accurately classifying various sports types based on their visual characteristics in images. This initiative addresses a challenging multi-class image classification problem involving **100 distinct sports categories**. The model leverages the power of Transfer Learning with a pre-trained Convolutional Neural Network (CNN) to achieve high accuracy on a diverse collection of sports images.

## Key Features & Methodologies

* **Transfer Learning:** Utilizes the MobileNetV2 architecture, pre-trained on the extensive ImageNet dataset, as a powerful feature extractor. The base layers of MobileNetV2 are frozen to preserve its learned hierarchical features, enabling efficient training with our specific sports image dataset.

* **Data Augmentation:** Real-time data augmentation techniques (e.g., rotation, zoom, horizontal flip) are applied to the training dataset. This significantly enhances the model's generalization capability and robustness to unseen image variations by effectively increasing the diversity of the training data.

* **Optimized Training:** The model is trained using the Adam optimizer with a controlled learning rate, ensuring stable and precise convergence.

* **EarlyStopping:** Prevents overfitting by monitoring the validation loss and automatically halting training when no significant improvement is observed over a specified number of epochs.
   
* **Comprehensive Evaluation:** Model performance is rigorously evaluated on independent validation and test datasets using key metrics such as accuracy, precision, recall, F1-score, and a detailed confusion matrix.

* **Prediction Visualization:** Random samples from the test set are visualized with their true and predicted labels, offering a qualitative assessment of the model's classification strengths and weaknesses.

## Dataset

The dataset used for this project comprises a diverse collection of images categorized into 100 distinct sports types. This dataset serves as a rich foundation for training and evaluating the deep learning model. [Dataset](https://www.kaggle.com/datasets/gpiosenka/sports-classification)

## Results

The developed deep learning model has achieved an exceptional **94.6% accuracy rate** on the independent test dataset. This outstanding performance highlights the model's strong generalization ability and its reliability in classifying sports images it has never encountered during training. The high accuracy, combined with strong precision and recall scores across numerous classes, confirms the effectiveness of the chosen Transfer Learning approach and optimization strategies for this complex multi-class problem.

## Technologies Used

* **Python 3.x**
* **TensorFlow / Keras:** For building, training, and evaluating the deep learning model.
* **MobileNetV2:** As the pre-trained convolutional base for Transfer Learning.
* **NumPy:** For numerical operations and data manipulation.
* **Matplotlib:** For data visualization, including training history plots and confusion matrices.
* **Seaborn:** For enhanced data visualization, particularly for the confusion matrix heatmap.
* **Scikit-learn:** For classification metrics (e.g., `classification_report`, `confusion_matrix`).