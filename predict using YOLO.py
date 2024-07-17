import os
import cv2
from ultralytics import YOLO

IMAGES_DIR = os.path.join('.', 'test')
image_path = os.path.join(IMAGES_DIR, 'weed1.jpg')
image_out_path = '{}_out.jpg'.format(os.path.splitext(image_path)[0])

# Load the image
image = cv2.imread(image_path)

# Load the model
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')
model = YOLO(model_path)  # Load a custom model

threshold = 0.5

# Perform object detection
results = model(image)[0]

# Annotate the image
for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            if int(class_id)==0:
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1)+35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            else:
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1)+35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
                 

# Save the annotated image
cv2.imwrite(image_out_path, image)

print("Annotated image saved at:", image_out_path)
