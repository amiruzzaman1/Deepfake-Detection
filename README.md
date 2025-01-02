# Deepfake-Detection

This project focuses on deepfake detection leveraging the OpenForensics dataset, a comprehensive resource for face forgery detection and segmentation research. The project explores various deep learning models and evaluates their performance in distinguishing between real and fake images of human faces.

## Methodology

### A. Dataset
**Dataset Link:** [Emotions Dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)

The OpenForensics dataset contains 256x256 jpg images categorized into real and fake human faces:
- **Train:** 140,000 images (70,000 real, 70,000 fake)
- **Test:** 10,905 images (5,492 fake, 5,413 real)
- **Validation:** 39,400 images (19,600 fake, 19,800 real)
This structured dataset provides a diverse range of manipulated and authentic faces, challenging deep learning models to detect subtle differences and enhance research in deepfake detection.

### B. Pre-processing
Data preprocessing involves:
- **Transformations:** Resizing, rotation, and normalization of images.
- **Class Balancing:** Random oversampling to address class imbalance.
- **Dataset Split:** 80-20 split for training and testing sets.
- **Pre-trained Weights:** Initialization for transfer learning with Vision Transformer (ViT).

### C. Models
Five models are evaluated:
1. **Vision Transformer (ViT):** Effective for image classification with strong performance.
2. **MobileNetV2:** Lightweight architecture suitable for mobile applications.
3. **Convolutional Neural Network (CNN):** Known for image recognition tasks.
4. **ResNet50:** Deep residual learning for feature extraction.
5. **EfficientNetB0:** Efficient model scaling for resource optimization.

### D. Evaluation
Models are evaluated based on:
- **Accuracy:** Training and validation accuracy metrics.
- **Loss:** Training and validation loss metrics.

## Results

### Model Performance

| Model           | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
|-----------------|-------------------|---------------------|---------------|-----------------|
| ViT             | 99%               | 99%                 | 0.04          | 0.02            |
| MobileNetV2     | 99%               | 98%                 | 0.02          | 0.05            |
| CNN             | 97%               | 95%                 | 0.05          | 0.12            |
| ResNet50        | 84%               | 83%                 | 0.35          | 0.36            |
| EfficientNetB0  | 50%               | 50%                 | 0.69          | 0.69            |

### Model Insights
- **Vision Transformer (ViT):** Achieved high accuracy and low loss, demonstrating robust performance in distinguishing between real and fake images.
- **MobileNetV2:** Lightweight and efficient, suitable for real-time applications with strong accuracy.
- **CNN:** Effective but showed higher validation loss, indicating potential overfitting.
- **ResNet50:** Moderate accuracy with higher loss compared to ViT and MobileNetV2, suggesting room for optimization.
- **EfficientNetB0:** Struggled with low accuracy and high loss, indicating challenges in model fitting for this task.

## Streamlit GUI
[Link to Deepfake Image Classification](https://huggingface.co/spaces/Amiruzzaman/Deepfake_Image_Classification)

## Sample 
![image](https://github.com/user-attachments/assets/8754fc57-ece2-4239-9140-9fcec27234d4)

![image](https://github.com/user-attachments/assets/261fb2e6-b782-4649-be5a-6fa6aa873291)

## Link to the Application
[Link to Deepfake Image Classification](https://huggingface.co/spaces/Amiruzzaman/Deepfake_Image_Classification)


