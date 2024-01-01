from configparser import ConfigParser

config = ConfigParser()

config["DETR"] = {
    "ConfidenceThreshold" : 0.87,
    "NMS_IOU_Threshold" : 0.6,
    "Images_folder_path" : "/home/ailearner/internship/Testset_3231images",
    "GT_annotations_folder_path" : "/home/ailearner/internship/persondetection/cocofileGT/gt.json",
    "DT_folder_path" : "/home/ailearner/internship/automationofEval/detectionsdetr-resnet-101",
    "modelWeights_path" : "facebook/detr-resnet-101",
    "Evaluation_IOU_Threshold" : 0.25,
    "Results_folder_path" : "/home/ailearner/internship/EVOFresults",
}
config["YOLO_NAS"] = {
    "ConfidenceThreshold" : 0.6,
    "NMS_IOU_Threshold" : 0.5,
    "Images_folder_path" : "/home/ailearner/internship/Testset_3231images",
    "GT_annotations_folder_path" : "/home/ailearner/internship/GT",
    "DT_folder_path" : "/home/ailearner/internship/detections",
    "modelWeights_path" : "NO",
    "Evaluation_IOU_Threshold" : 0.25,
    "Results_folder_path" : "/home/ailearner/internship/results",
}

with open("config.ini", "w") as f:
    config.write(f)
