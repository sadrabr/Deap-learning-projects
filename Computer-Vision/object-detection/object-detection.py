import cv2
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSD300_VGG16_Weights

# Select device: GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained SSD model with VGG16 backbone
weights = SSD300_VGG16_Weights.DEFAULT
model = ssd300_vgg16(weights=weights).to(device)
model.eval()

# Load the preprocessing transforms and class labels
transform = weights.transforms()
class_names = weights.meta['categories']

# Load video
video_path = 'funny_dog.mp4'  # Replace with your actual video file name
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# Process video frame-by-frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert OpenCV BGR image to RGB, then to PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Apply transform and move to device
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Object detection
    with torch.no_grad():
        outputs = model(input_tensor)[0]

    # Draw bounding boxes and labels
    for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box.tolist())
            class_name = class_names[label]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name}: {score:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Write and display the frame
    out.write(frame)
    cv2.imshow('SSD Object Detection', frame)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
