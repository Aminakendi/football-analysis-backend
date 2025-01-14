import os
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO('models/object.pt')

# Path to the data.yaml file
data_yaml_path = r'C:\Users\Eric\Desktop\project_main\football-players-detection-2\data.yaml'

# Evaluate the model on the test set
results = model.val(data=data_yaml_path, split='test', save_json=True)


# Access overall evaluation metrics
print("\nPerformance Metrics:")
metrics = results.results_dict
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Summarize detections (directly from results)
print("\nClass-Specific Metrics:")
for idx, cls_name in enumerate(results.names):
    class_result = results.class_result(idx)
    print(f"Class: {cls_name}")
    print(f" Precision: {class_result[0]:.4f}")
    print(f" Recall: {class_result[1]:.4f}")
    print(f" mAP@50: {class_result[2]:.4f}")
    print(f" mAP@[50:95]: {class_result[3]:.4f}")
