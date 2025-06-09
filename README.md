# 🐾 Animal Faces Classification with PyTorch

This project uses a Convolutional Neural Network (CNN) built with **PyTorch** to classify animal face images from the [Animal Faces dataset](https://www.kaggle.com/datasets/andrewmvd/animal-faces). The dataset contains three classes: **cat**, **dog**, and **wild**.

## 📁 Dataset

- Source: [Kaggle - Animal Faces](https://www.kaggle.com/datasets/andrewmvd/animal-faces)
- Structure:
/animal-faces/afhq/
├── train/
│ ├── cat/
│ ├── dog/
│ └── wild/
├── val/
│ ├── cat/
│ ├── dog/
│ └── wild/
└── test/
├── cat/
├── dog/
└── wild/

perl
Copy
Edit

## ⚙️ Requirements

Install the required packages:

```bash
pip install opendatasets torch torchvision scikit-learn matplotlib pandas pillow
Also, you'll need Kaggle API credentials to download the dataset:

Go to Kaggle Account.

Click on Create New API Token.

Upload your kaggle.json or provide the username and key manually.

📥 Download Dataset
python
Copy
Edit
import opendatasets as od
od.download("https://www.kaggle.com/datasets/andrewmvd/animal-faces")
🧠 Model Architecture
A custom CNN model with the following layers:

Conv2D (3 → 32)

MaxPool

ReLU

Conv2D (32 → 64)

MaxPool

ReLU

Conv2D (64 → 128)

MaxPool

ReLU

Flatten

Dense (128 units)

Output Layer (3 classes)

🔁 Data Pipeline
Resize images to 128x128

Normalize and convert to tensors

Encode labels (cat/dog/wild → 0/1/2)

python
Copy
Edit
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])
📊 Training & Validation
Loss: CrossEntropyLoss

Optimizer: Adam

Epochs: 10

Batch Size: 16

Validation Split: 15%

Accuracy and loss are printed and logged for each epoch.

python
Copy
Edit
for epoch in range(EPOCHS):
    ...
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Acc: ... - Val Acc: ...")
📈 Results
During training, both accuracy and loss are recorded for:

Training set

Validation set

You can visualize the performance using matplotlib:

python
Copy
Edit
plt.plot(total_acc_train_plot, label="Train Accuracy")
plt.plot(total_acc_validation_plot, label="Val Accuracy")
...
🧪 Testing
Evaluate the model on unseen test data:

python
Copy
Edit
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        ...
📷 Sample Images
Below is a 3x3 grid of randomly sampled images from the dataset:


💾 Model Summary
markdown
Copy
Edit
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 128, 128]             896
         MaxPool2d-2           [-1, 32, 64, 64]               0
            ...
           Linear-12                    [-1, 3]             387
================================================================
Total params: 4,288,067
Trainable params: 4,288,067
----------------------------------------------------------------
🧠 Future Improvements
Use Transfer Learning with pretrained CNNs (e.g., ResNet18)

Implement real-time webcam classification

Add confusion matrix and precision/recall

📬 Contact
Made with ❤️ by Anas
📧 Email: anasbkmuhaisen@gmail.com
