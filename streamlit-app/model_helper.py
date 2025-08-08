from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

trained_model= None
class_names = ['F_Breakage', 'F_Crushed', 'F_Normal', 'R_Breakage', 'R_Crushed', 'R_Normal']

# Load the pre-trained ResNet model
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        # Freeze all layers except the final fully connected layer

        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layer4 and fc layers
        for param in self.model.layer4.parameters():
            param.requires_grad = True

            # Replace the final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x


def predict(image_path):
    image =Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # reduce image
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    global trained_model
    image_tensor = transform(image).unsqueeze(0)
    if trained_model is None:
        image_tensor = transform(image).unsqueeze(0) #(3(rgb),224,224 (image size)) convert batches
        trained_model = CarClassifierResNet()
        trained_model.load_state_dict(torch.load(r"model\saved_model.pth"))
        trained_model.eval()

    with torch.no_grad():
        output = trained_model(image_tensor) #[[12,14,9,15,8,20]]
        _,predicted_class = torch.max(output,1) #20 5 output (higher number, index)
        return class_names[predicted_class.item()]
