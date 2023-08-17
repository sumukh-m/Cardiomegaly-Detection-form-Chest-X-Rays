
# Cardiomegaly Detection using Chest X-Rays using CNN and Transfer Learning

The "Cardiomegaly Detection using Chest X-Rays using CNN and Transfer Learning" project aims to develop a robust and accurate automated system for detecting cardiomegaly, a condition characterized by an enlarged heart, by analyzing chest X-ray images. The project leverages Convolutional Neural Networks (CNNs) and employs transfer learning techniques using popular pre-trained models such as ResNet50, InceptionV3, DenseNet121, and EfficientNetB0.

Cardiomegaly is a significant indicator of various cardiovascular diseases and can be identified through radiological imaging. However, manual interpretation of chest X-rays is time-consuming and prone to human error. The proposed system seeks to address this issue by providing an automated solution that can quickly and accurately identify signs of cardiomegaly, thereby assisting medical professionals in making timely and informed decisions regarding patient care.


F1-Score
<pre>
                ResNet152V2  InceptionV3  MobileNetV1  EfficientNetB4
Balanced            79%          78%          81%           77%
Unbalanced          82%          81%          83%           81%

</pre>

# Implementation
The implementation of our Cardiomegaly Detection system was done using the Keras deep learning framework with a TensorFlow backend. We used Python as our programming language. We used the publicly available NIH (National Institute of Health) Chest X-ray dataset, which consists of 112,120 frontal-view chest X-ray images from 30,805 unique patients, labeled with 14 different thoracic diseases, including cardiomegaly. We preprocessed the images by resizing them to respective input image sizes of models and normalizing the pixel values to a range of [0,1]. We first organized the Chest X-ray dataset into two categories: one for cases with cardiomegaly and one for cases without cardiomegaly. We then performed exploratory data analysis techniques to examine the distributions and check the balance of the dataset.

We split the dataset into training, validation, and test datasets with a ratio of 80:08:12. To balance the distribution in the dataset, we used oversampling to increase the number of cases with cardiomegaly by data augmentation. We used the ImageDataGenerator function in Keras to perform data augmentation on the training dataset, including horizontal and vertical flipping, random rotation, and zooming.

The implementation of the dataset on the entire model without any pretrained weights, as well as on the model with an appended layer of fully connected layers, did not give satisfactory accuracy. This approach was tried initially to train the model from scratch without any prior knowledge or weights. However, due to the lack of training data and the complexity of the problem, the accuracy obtained was not up to the mark. Hence, the transfer learning approach was utilized, where a pre-trained model was used as a starting point for training the model on the given dataset. Although the accuracy improved significantly, the performance was still not optimal.

Therefore, we added an attention model to the pre-trained models to improve their performance. We also applied global weighted average pooling to capture the most important features of the images and output them to a classification layer with dropout and two fully connected layers.

We also applied these techniques on both unbalanced and balanced datasets. Using an unbalanced dataset can be problematic because it can lead to biased and inaccurate model performance. In an unbalanced dataset, the number of examples in each class is not evenly distributed. This means that the model is more likely to be trained on one class over another, which can result in a biased model that performs poorly on the underrepresented class. On the other hand, a balanced dataset ensures that the model is trained on an equal number of examples from each class, resulting in a more accurate and unbiased model. The model is forced to learn the features that are common to all classes rather than relying on the bias towards one class.

We trained the models using transfer learning, fine-tuning pre-trained models on the ImageNet dataset, including Efficient Net B4, InceptionV3, MobileNetV1, and ResNet152-V2. We initialized the models with their pre-trained weights and then trained them on the balanced NIH dataset with the added attention model and global weighted average pooling. We used the binary cross-entropy loss function and the Adam optimizer with a learning rate of .001. We used early stopping with patience of 10 to prevent overfitting.

![image](https://github.com/sumukh-m/Cardiomegaly-Detection/blob/master/Screenshots/Custom%20Model.png)

# Results and Evaluation

![image]()
