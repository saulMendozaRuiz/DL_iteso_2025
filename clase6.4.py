import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
from PIL import Image

# Function to classify webcam image
def classify_webcam_image(model, device, transform):
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the frame to a PIL Image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Preprocess the image
        img = transform(img).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            model.eval()
            output = model(img)
            pred = output.argmax(dim=1).item()

        # Display the frame with the prediction
        cv2.putText(frame, f'Predicted Class: {pred}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Webcam', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Define device (use GPU if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load pre-trained ResNet50 model and modify the final layer
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)  # STL-10 has 10 classes
    model.load_state_dict(torch.load(r'models\custom_resnet.pth'))
    model = model.to(device)

    # Classify images from the webcam
    classify_webcam_image(model, device, transform)