from PIL import Image, ImageOps
from model.card_detect import CardDetectModel
from model.info_extract import InfoExtractModel
from model.post_processing import PostProcessing


class Model:
    def __init__(self):
        self.card_model = None
        self.info_model = None
        self.post_processing = None
        
    
    def load_model(self):
        path_weight_corner = "weights_file/weight_corner.pt"
        path_weight_info = "weights_file/weight_extract_info.pt"
        self.card_model = CardDetectModel(path_weight_corner)
        self.info_model = InfoExtractModel(path_weight_info)
        self.post_processing = PostProcessing()


    def predict(self, image_path):
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        num_of_corner, crop_img = self.card_model.crop_card(img)
        data = self.info_model.info_predict(crop_img)
        return self.post_processing.export_json(num_of_corner, crop_img, data)


