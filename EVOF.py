from configparser import ConfigParser
import os
config = ConfigParser()
config.read("config.ini")

modelname = input("what is your model name? ")
config_data = config[modelname]


def print_results(data_path, labels_path, detects_path, iou_threshold,results_path,confidence,nms,modelname ):
    print("Evaluation in progress...")
    iou_threshold = float(iou_threshold)
    confidence = float(confidence)
    nms = float(nms)
    
    import plotly.io as pio
    import fiftyone as fo
    import fiftyone.utils.coco as fouc

    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=data_path,
        labels_path=labels_path,
        label_types=["detections"],
        label_field="ground_truth",
    )

    classes = ["person"]

    fouc.add_coco_labels(
        dataset,
        "predictions",
        detects_path,
        classes,
    )

    results = dataset.evaluate_detections(
        "predictions",
        gt_field="ground_truth",
        eval_key="eval",
        # method="coco",
        # classes = classes,
        iou=iou_threshold,
        compute_mAP= True,
    )

    # Convert variables to strings with appropriate formatting
    confidence_str = "{:.2f}".format(confidence)
    var1_str = "nms_{}".format(nms)
    var2_str = "iou_{}".format(iou_threshold)
    var3_str = "model_{}".format(modelname)
    # Specify the nested folder path
    nested_folder_path = os.path.join(results_path, "results_{}_{}_{}_{}".format(confidence_str, var1_str, var2_str, var3_str))

    # Create the nested folder if it doesn't exist
    os.makedirs(nested_folder_path, exist_ok=True)

    # Specify the file path with the dynamic filename
    file_path = os.path.join(nested_folder_path, "results_{}_{}_{}_{}.txt".format(confidence_str, var1_str, var2_str, var3_str))

    # Open the file in write mode and write the results
    with open(file_path, 'w') as file:
        file.write("mAP: {}\n".format(results.mAP()))
        file.write("TP: %d\n" % dataset.sum("eval_tp"))
        file.write("FP: %d\n" % dataset.sum("eval_fp"))
        file.write("FN: %d\n" % dataset.sum("eval_fn"))
    
    plot = results.plot_pr_curves(classes=classes)
    
    # Save PR curve image
    pr_curve_image_path = os.path.join(nested_folder_path, f"pr_curve_{confidence_str}_{var1_str}_{var2_str}.png")

# Save the Plotly Figure as an image
    pio.write_image(plot, pr_curve_image_path)
    # Print a message indicating successful write
    print("Results written to {}".format(file_path))

def DETR(config_data):
    import json
    import os
    import cv2
    import torch
    import supervision as sv
    import transformers
    from transformers import DetrForObjectDetection, DetrImageProcessor

    # Set your configuration
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    CHECKPOINT = config_data["modelWeights_path"]
    #here model path is facebook/detr-resnet-101
    #but modelname should be detr-resnet-101
    modelname = config_data["modelWeights_path"].split("/")[1]
    # resnet-101-dc5
    CONFIDENCE_THRESHOLD = float(config_data["ConfidenceThreshold"])
    IOU_THRESHOLD = float(config_data["NMS_IOU_Threshold"])
    INPUT_FOLDER = config_data["Images_folder_path"]
    OUTPUT_FOLDER = config_data["DT_folder_path"]

    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Initialize the model and processor
    image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
    model = DetrForObjectDetection.from_pretrained(CHECKPOINT)
    model.to(DEVICE)
    # Initialize COCO dataset dictionary
    coco_dataset = {
        "info": {},
        "licenses": [],
        "categories": [{"id": 0, "name": "person"}],  # Assuming "person" is the desired class
        "images": [],
        "annotations": []
    }
    counter  = 0
    print("sit back and relax, it will take some time")
    # Process images in the input folder
    for filename in sorted(os.listdir(INPUT_FOLDER)):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            input_path = os.path.join(INPUT_FOLDER, filename)
            image = cv2.imread(input_path)
            inputs = image_processor(images=image, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)
            target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
            results = image_processor.post_process_object_detection(outputs=outputs, threshold=CONFIDENCE_THRESHOLD, target_sizes=target_sizes)[0]
            detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=IOU_THRESHOLD)
            
            image_id = int(''.join(filter(str.isdigit, filename.split('.')[0])))
            
            
            image_info = {
                "id": image_id,
                "width": image.shape[1],
                "height": image.shape[0],
                "file_name": filename
            }
            coco_dataset["images"].append(image_info)
            
            # counter= counter+1
            # if counter == 10:
            #     break
            
            for i in range(len(detections.xyxy)):
                bbox = detections.xyxy[i]
                confidence = round(detections.confidence[i].item(), 4)
                class_id = detections.class_id[i]
                class_label = model.config.id2label[class_id.item()]

                # Check if the class is the one you want to include in COCO format
                if class_label.lower() == "person":

                    x1, y1, x2, y2 = map(int, bbox.tolist())
                    # Add annotation information to COCO dataset
                    ann_info = {
                        "id": len(coco_dataset["annotations"]) + 1,
                        "image_id": image_id,
                        "category_id": 0,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO format: [x, y, width, height]
                        "area": (x2 - x1) * (y2 - y1),
                        "iscrowd": 0,
                        "score": confidence
                    }
                    coco_dataset["annotations"].append(ann_info)

    # Save the COCO dataset to a JSON file
    detections_file_name = "conf_{}_nms_{}_iou_{}_model_{}.json".format(config_data["ConfidenceThreshold"], config_data["NMS_IOU_Threshold"], config_data["Evaluation_IOU_Threshold"], modelname)
    output_json_path = os.path.join(OUTPUT_FOLDER, detections_file_name)
    with open(output_json_path, 'w') as json_file:
        json.dump(coco_dataset, json_file)

    print("Object detection and saving in COCO format complete.")
    print_results(config_data["Images_folder_path"], config_data["GT_annotations_folder_path"], output_json_path, config_data["Evaluation_IOU_Threshold"], config_data["Results_folder_path"], config_data["ConfidenceThreshold"], config_data["NMS_IOU_Threshold"], modelname)

if modelname == "DETR":
    modelname = config_data["modelWeights_path"].split("/")[1]
    #write a case of checking inside detection folder json file is there are not
    detections_file_name = "conf_{}_nms_{}_iou_{}_model_{}.json".format(config_data["ConfidenceThreshold"], config_data["NMS_IOU_Threshold"], config_data["Evaluation_IOU_Threshold"], modelname)
    
    detections_file_path = os.path.join(config_data["DT_folder_path"], detections_file_name)
    
    
    if os.path.exists(detections_file_path) and os.path.isfile(detections_file_path):
        print("found the detections file")
        print_results(config_data["Images_folder_path"], config_data["GT_annotations_folder_path"], detections_file_path, config_data["Evaluation_IOU_Threshold"], config_data["Results_folder_path"], config_data["ConfidenceThreshold"], config_data["NMS_IOU_Threshold"], modelname)
    else:
        print("detections file not found")
        DETR(config_data)
        
        
        
        
        
        
    
    