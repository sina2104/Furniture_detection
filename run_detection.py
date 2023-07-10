import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image



def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = get_model_instance_segmentation(4)
model.load_state_dict(torch.load('furniture_detection_model.pth'))
model.to(device)
model.eval()

image_path = r"Test_images/Test_image3.png"
image = Image.open(image_path).convert('RGB')

# Apply transformations to the image
transform = get_transform()
image_tensor = transform(image).to(device)

# Reshape the image tensor to include batch dimension
image_tensor = image_tensor.unsqueeze(0)

# Run the image through the model
model.eval()
with torch.no_grad():
    predictions = model(image_tensor)

# Process the predictions
boxes = predictions[0]['boxes'].detach().cpu().numpy()
labels = predictions[0]['labels'].detach().cpu().numpy()
scores = predictions[0]['scores'].detach().cpu().numpy()

# Filter out predictions with low scores
threshold = 0.5
filtered_indices = scores >= threshold
filtered_boxes = boxes[filtered_indices]
filtered_labels = labels[filtered_indices]
filtered_scores = scores[filtered_indices]

# Set th name fo each class label
class_names = {
    1: 'Chair',
    2: 'Sofa',
    3: 'Table'
 }

# Set a color for each class label
class_colors = {
    1: (0, 255, 0),   # Green color chair
    2: (0, 0, 255),   # Red color for sofa
    3: (255, 0, 0)    # Blue color for table
}

# Print the filtered bounding boxes and labels
for box, label, score in zip(filtered_boxes, filtered_labels, filtered_scores):
    print('Label:', label)
    print('Score:', score)
    print('Bounding Box:', box)

# Draw bounding boxes on the image
image = cv2.imread(image_path)
for box, label, score in zip(boxes, labels, scores):
    if score >= threshold:
        # Extract coordinates
        xmin, ymin, xmax, ymax = box
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

        # Get class color
        color = class_colors[label]
        label_name = class_names[label]
        # Draw the bounding box rectangle and label text
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image, f'{label_name} {"%.2f" % score}' , (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Save the image with bounding boxes
cv2.imwrite('Test_images/Test_result3.jpg', image)