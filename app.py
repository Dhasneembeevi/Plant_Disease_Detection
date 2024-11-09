from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models


train_dir = r'D:\GUVI\CODE\Plant_Diseases_Detection\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train'
valid_dir = r'D:\GUVI\CODE\Plant_Diseases_Detection\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid'


train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),                
    transforms.RandomHorizontalFlip(),             
    transforms.ColorJitter(brightness=0.2, contrast=0.2), 
    transforms.ToTensor(),                          
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])

valid_transforms = transforms.Compose([
    transforms.Resize((64, 64)),                
    transforms.ToTensor(),                         
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
])

train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(root=valid_dir, transform=valid_transforms)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = nn.ReLU()(self.fc1(x))
        return self.fc2(x)

model = models.resnet18(weights=None)
num_classes = len(train_dataset.classes)  
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(r"D:\GUVI\CODE\Plant_Diseases_Detection\best_model.pth", map_location=torch.device('cpu')))
model.eval() 


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class_names = train_dataset.classes


st.title("Plant Disease Detection")
st.write("Upload a plant leaf image to detect disease")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    
    image_tensor = transform(image).unsqueeze(0) 

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs.data, 1)
        predicted_class_name = class_names[predicted_idx.item()]

    

    # prediction_cleaned = predicted_class_name.split("___")[-1]


    # formatted_prediction = prediction_cleaned.replace("_", " ") + " disease"

    print("Prediction:", predicted_class_name)
    st.write(f"Prediction: **{predicted_class_name}**")