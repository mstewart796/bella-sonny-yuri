# CNN Project - First Attempt

This project represents my **first-ever implementation of a Convolutional Neural Network (CNN)**. It was a fundamental step in my journey into deep learning, where I built, trained, and tested a CNN for image classification using one of the great loves of my life - my cats! The code was written in Python using TensorFlow and Keras and written at the end of my Machine Learning A-Z course that can be bought on Udemy.

---

## **Overview**
The CNN is designed to classify images into **three categories**, using a dataset structured into training and testing sets. The model is trained on images processed with data augmentation and then validated using a test dataset. After training, it predicts the class of a single test image.

---

## **Dataset**
- The training and testing datasets are stored in structured directories.
- The training data undergoes preprocessing, including:
  - **Rescaling**: Normalizing pixel values to the range [0, 1].
  - **Augmentation**: Applying shear, zoom, and horizontal flip transformations to improve generalization.
- The test data is only rescaled for consistency during validation.

---

## **Model Architecture**
The CNN follows a straightforward sequential architecture:
1. **Convolutional Layers**:
   - Two layers, each with 32 filters and ReLU activation.
   - Extract features like edges and textures.
2. **Max Pooling Layers**:
   - Two layers with a pooling size of 2x2.
   - Reduce spatial dimensions to prevent overfitting and improve computational efficiency.
3. **Fully Connected Layers**:
   - A flattening layer followed by:
     - Dense layer with 128 units and ReLU activation.
     - Output layer with 3 units (for the three classes) and softmax activation.

---

## **Strengths**
- **Data Augmentation**: I Implemented transformations like zoom, shear, and horizontal flips improves the robustness of the model.
- **Straightforward Architecture**: Being a begineer, the CNN's structure was kept simple and as such it is well-suited for other beginners as it is easy to understand and debug.
- **Efficient Training**: The model uses the Adam optimizer, which is super popular and great for newbies like me.

---

## **Weaknesses**
1. **Limited Dataset Information**:
   - It was a tiny dataset, I have literally hundreds of photos of my cats (if you look at the gallery on my phone it is 90% cats) but very few of them individually as they are normally rogether.
2. **Image Resolution**:
   - Images are resized to 64x64, which may lead to loss of detail.
3. **Model Complexity**:
   - Using only two convolutional layers may limit the model's ability to capture complex patterns.
4. **Overfitting Risk**:
   - With 25 epochs and relatively simple architecture, the model might overfit if the dataset is small or lacks diversity.
5. **Prediction Feedback**:
   - The script predicts only one image, with no systematic evaluation of accuracy on unseen data beyond the test set.

---

## **Improvements to Consider**
1. **Enhanced Dataset**:
   - Use a larger, more diverse dataset to improve the model's robustness.
2. **Additional Layers**:
   - Add more convolutional and pooling layers to capture deeper features.
3. **Regularization Techniques**:
   - Include dropout layers to reduce overfitting.
4. **Image Resolution**:
   - Experiment with higher resolutions (e.g., 128x128) for better feature extraction.
5. **Evaluation Metrics**:
   - Implement precision, recall, and F1-score to assess the model's performance in greater detail.
6. **Automated Evaluation**:
   - Use a batch of unseen images for predictions to test the model's practical utility.

---

## **Reflection**
Building this CNN was a significant milestone in my learning journey. While it has room for improvement, it gave me invaluable insights into deep learning workflows, data preprocessing, and model training. I look forward to enhancing my skills by tackling more complex architectures and datasets in the future. 

---

## **How to Use**
1. Ensure the dataset is organized with `training_set` and `test_set` directories.
2. Install dependencies: `TensorFlow`, `Keras`, `SciPy`, and `NumPy`.
3. Run the script to train the model.
4. Use the provided test image example or your own image to make predictions.

---

Thank you for reviewing my first attempt! Constructive feedback is always welcome. 😊
