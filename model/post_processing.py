from PIL import ImageEnhance
import json
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


class PostProcessing:
    def __init__(self):
        # Load vietocr
        config = Cfg.load_config_from_name('vgg_seq2seq')
        # config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
        config['cnn']['pretrained'] = False
        config['device'] = 'cpu'
        config['predictor']['beamsearch'] = False
        self.ocr = Predictor(config)


    def crop_info(self, img, list_box):
        crop_img = img.crop((list_box[0]["xmin"], list_box[0]["ymin"]-5, list_box[len(
            list_box)-1]["xmax"], list_box[len(list_box)-1]["ymax"]+5))
        enhancer = ImageEnhance.Sharpness(crop_img)
        factor = 2
        crop_img = enhancer.enhance(factor)
        return crop_img


    def crop_info_one_box(self, img, box):
        crop_img = img.crop(
            (box["xmin"], box["ymin"], box["xmax"], box["ymax"]))
        enhancer = ImageEnhance.Sharpness(crop_img)
        factor = 2
        crop_img = enhancer.enhance(factor)
        return crop_img


    def extract_info(self, img):
        result = self.ocr.predict(img)
        return result


    def combine_word(self, num_of_corner, crop_img, list_box):
        if len(list_box) == 0:
            return "None"
        i = 0
        while i < len(list_box):
            count = 0
            for j in range(0, len(list_box)):
                if abs(list_box[i]["ymin"] - list_box[j]["ymin"]) > 70:
                    count += 1
            if count > 1:
                list_box.pop(i)
                i -= 1
            i += 1
        sort_x_list = sorted(list_box, key=lambda x: x["xmin"])
        if (sort_x_list[0]["name"] == "date"):
            res_str = ""
            for box in sort_x_list:
                res_str += self.extract_info(
                    self.crop_info_one_box(crop_img, box)) + "/"
            res_str = res_str[0:len(res_str) - 1]
            return res_str
        item_1 = sort_x_list[0]
        item_2 = "one_line"
        for data in sort_x_list:
            if (data["ymin"] + data["ymax"]) / 2 > item_1["ymax"] or (data["ymin"] + data["ymax"]) / 2 < item_1["ymin"]:
                item_2 = data
                break
        if item_2 == "one_line":
            if num_of_corner == 3:
                res_str = ""
                for box in sort_x_list:
                    res_str += self.extract_info(
                        self.crop_info_one_box(crop_img, box)) + " "
                return res_str
            return self.extract_info(self.crop_info(crop_img, sort_x_list))
        else:
            if (item_1["ymin"] + item_1["ymax"]) / 2 < item_2["ymin"]:
                up_item = item_1
                down_item = item_2
            else:
                up_item = item_2
                down_item = item_1
            list_up = []
            list_down = []
            for data in sort_x_list:
                if (data["ymin"] + data["ymax"])/2 > up_item["ymax"]:
                    list_down.append(data)
                else:
                    list_up.append(data)
            if num_of_corner == 3:
                res_str = ""
                for box in list_up:
                    res_str += self.extract_info(
                        self.crop_info_one_box(crop_img, box)) + " "
                for box in list_down:
                    res_str += self.extract_info(
                        self.crop_info_one_box(crop_img, box)) + " "
                return res_str
            return self.extract_info(self.crop_info(crop_img, list_up)) + " " + self.extract_info(self.crop_info(crop_img, list_down))


    def export_json(self, num_of_corner, crop_img, data):
        id_data = []
        name_data = []
        birth_data = []
        home_data = []
        add_data = []
        date_data = []
        place_data = []
        for info in data:
            if info["name"] == "id":
                id_data.append(info)
            elif info["name"] == "name":
                name_data.append(info)
            elif info["name"] == "birth":
                birth_data.append(info)
            elif info["name"] == "home":
                home_data.append(info)
            elif info["name"] == "add":
                add_data.append(info)
            elif info["name"] == "date":
                date_data.append(info)
            elif info["name"] == "place":
                place_data.append(info)
        json_data = {
            "id": self.combine_word(num_of_corner, crop_img, id_data),
            "name": self.combine_word(num_of_corner, crop_img, name_data),
            "birth": self.combine_word(num_of_corner, crop_img, birth_data),
            "home": self.combine_word(num_of_corner, crop_img, home_data),
            "add": self.combine_word(num_of_corner, crop_img, add_data),
            "date": self.combine_word(num_of_corner, crop_img, date_data),
            "place": self.combine_word(num_of_corner, crop_img, place_data)
        }
        return json.dumps(json_data, ensure_ascii=False)
