This project applies Image & Video Processing (IVP) techniques to MRI brain scans to improve contrast and visibility before feeding them into a Convolutional Neural Network (CNN) for binary classification:

Tumor

No Tumor

IVP preprocessing dramatically improves the image quality and helps the model learn features more effectively.

🗂️ Dataset

Kaggle Dataset: Brain MRI Images for Tumor Detection

Folder structure:

brain_tumor_dataset/
│── yes/      # Tumor images (~155)
│── no/       # Non-tumor images (~98)


Total images ≈ 253.

🔧 Technologies & Libraries Used
Image Processing

OpenCV (cv2)

NumPy

Matplotlib

Deep Learning

TensorFlow / Keras

scikit-learn (train/test split, metrics)

Visualization

Matplotlib

Seaborn

🎞️ IVP Techniques Applied
1️⃣ Grayscale Conversion

Simplifies computation and retains structural intensity.

2️⃣ Logarithmic Transform

Enhances dark regions.

3️⃣ Gamma Correction

Adjusts brightness (γ < 1 brightens, γ > 1 darkens).

4️⃣ Histogram Equalization

Improves global contrast.

5️⃣ CLAHE (Contrast Limited Adaptive Histogram Equalization)

Best performer for MRI images.
Avoids noise amplification and boosts local contrast.

6️⃣ Canny Edge Detection

Highlights edges and possible tumor boundaries.

🧪 Model Architecture (CNN)
Conv2D(32, 3×3) + ReLU
MaxPooling2D(2×2)

Conv2D(64, 3×3) + ReLU
MaxPooling2D(2×2)

Conv2D(128, 3×3) + ReLU
MaxPooling2D(2×2)

Flatten
Dense(128) + ReLU
Dropout(0.3)

Dense(1) + Sigmoid


Loss: Binary Crossentropy

Optimizer: Adam

Activation: Sigmoid

Metric: Accuracy

EarlyStopping: Enabled to avoid overfitting

📊 Training Pipeline

Load images (yes/no)

Convert to grayscale

Resize to 128×128

Apply CLAHE

Normalize to [0,1]

Expand shape → (N,128,128,1)

Split dataset (80% train, 20% validation)

Train CNN with EarlyStopping

Evaluate using metrics

📈 Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

ROC-AUC

Confusion Matrix

CLAHE preprocessing gave the best improvements in prediction performance and visual clarity.

▶️ Demo Prediction

A function demo_predict(path) allows you to test the model on any individual MRI image:

demo_predict("brain_tumor_dataset/yes/Y1.jpg")


Displays the image and prints:
Tumor or No Tumor + probability.

🧾 Project Report

A full detailed PDF report is included:

Brain_Tumor_Detection_IVP_Report_Rijul_Niketan_Vishal.pdf

🚀 Future Improvements

Use deeper CNNs / Transfer Learning (VGG16, ResNet50)

Add data augmentation

Implement Grad-CAM visualizations

Extend to multi-class tumor detection

Create segmentation model (e.g., U-Net)

🤝 Acknowledgements

Gonzalez & Woods — Digital Image Processing

Kaggle MRI Tumor Dataset

TensorFlow Documentation

OpenCV Documentation
