import torch
import json


class InfoExtractModel:
    def __init__(self, path_weight):
        self.model = torch.hub.load(
            'ultralytics/yolov5', 'custom', path=path_weight, force_reload=True)


    def IoU(self, box1, box2):
        # Tinh dien tich 2 box
        area1 = (box1["xmax"] - box1["xmin"]) * (box1["ymax"] - box1["ymin"])
        area2 = (box2["xmax"] - box2["xmin"]) * (box2["ymax"] - box2["ymin"])

        # Tim toa do giao nhau
        xx = max(box1["xmin"], box2["xmin"])
        yy = max(box1["ymin"], box2["ymin"])
        aa = min(box1["xmax"], box2["xmax"])
        bb = min(box1["ymax"], box2["ymax"])

        # Tính diện tích vùng giao nhau
        w = max(0, aa - xx)
        h = max(0, bb - yy)
        intersection_area = w*h

        # Tính diện tích phần hợp nhau
        union_area = area1 + area2 - intersection_area
        # Dựa trên phần giao và phần hợp để tính IoU
        IoU = intersection_area / union_area

        return IoU


    def NMS(self, list_box, thresh_iou):
        confident_order = sorted(list_box, key=lambda x: x["confidence"])
        keeps = []
        while len(confident_order) > 0:
            cur_box = confident_order.pop(-1)
            keeps.append(cur_box)
            for index, box in enumerate(confident_order):
                if self.IoU(cur_box, box) > thresh_iou:
                    confident_order.pop(index)
        keeps = sorted(keeps, key=lambda x: x["ymax"])
        return keeps


    def info_predict(self, img):
        results = self.model(img)
        # results.show()
        result_list = json.loads(results.pandas().xyxy[0].sort_values(
            "ymax").to_json(orient="records"))
        return self.NMS(result_list, 0.2)


class InfoExtractModel:
    def __init__(self, path_weight):
        self.model = torch.hub.load(
            'ultralytics/yolov5', 'custom', path=path_weight, force_reload=True)


    def IoU(self, box1, box2):
        # Tinh dien tich 2 box
        area1 = (box1["xmax"] - box1["xmin"]) * (box1["ymax"] - box1["ymin"])
        area2 = (box2["xmax"] - box2["xmin"]) * (box2["ymax"] - box2["ymin"])

        # Tim toa do giao nhau
        xx = max(box1["xmin"], box2["xmin"])
        yy = max(box1["ymin"], box2["ymin"])
        aa = min(box1["xmax"], box2["xmax"])
        bb = min(box1["ymax"], box2["ymax"])

        # Tính diện tích vùng giao nhau
        w = max(0, aa - xx)
        h = max(0, bb - yy)
        intersection_area = w*h

        # Tính diện tích phần hợp nhau
        union_area = area1 + area2 - intersection_area
        # Dựa trên phần giao và phần hợp để tính IoU
        IoU = intersection_area / union_area

        return IoU


    def NMS(self, list_box, thresh_iou):
        confident_order = sorted(list_box, key=lambda x: x["confidence"])
        keeps = []
        while len(confident_order) > 0:
            cur_box = confident_order.pop(-1)
            keeps.append(cur_box)
            for index, box in enumerate(confident_order):
                if self.IoU(cur_box, box) > thresh_iou:
                    confident_order.pop(index)
        keeps = sorted(keeps, key=lambda x: x["ymax"])
        return keeps


    def info_predict(self, img):
        results = self.model(img)
        # results.show()
        result_list = json.loads(results.pandas().xyxy[0].sort_values(
            "ymax").to_json(orient="records"))
        return self.NMS(result_list, 0.2)
