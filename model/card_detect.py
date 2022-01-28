import torch
from PIL import Image
import json
import cv2
import numpy as np


class CardDetectModel:
    def __init__(self, path_weight):
        self.model = torch.hub.load(
            'ultralytics/yolov5', 'custom', path=path_weight)


    def corner_predict(self, img):
        results = self.model(img)
        return results.pandas().xyxy[0].sort_values("confidence").to_json(orient="records")


    def perspective_transform(self, image, source_points):
        dest_points = np.float32([[0, 0], [500, 0], [500, 300], [0, 300]])
        matrix = cv2.getPerspectiveTransform(source_points, dest_points)
        destination = cv2.warpPerspective(image, matrix, (500, 300))
        return destination


    def find_miss_corner(self, coordinate_dict):
        position_name = ['bottom_left',
                         'bottom_right', 'top_left', 'top_right', ]
        position_index = np.array([0, 0, 0, 0])

        for name in coordinate_dict.keys():
            if name in position_name:
                position_index[position_name.index(name)] = 1

        index = np.argmin(position_index)

        return index


    def calculate_missed_coord_corner(self, list_of_coordinate_dict):
        thresh = 0
        index = self.find_miss_corner(list_of_coordinate_dict)

        # calculate missed corner coordinate
        # case 1: missed corner is "top_left"
        if index == 2:
            midpoint = np.add(
                list_of_coordinate_dict['top_right'], list_of_coordinate_dict['bottom_left']) / 2
            y = 2 * midpoint[1] - \
                list_of_coordinate_dict['bottom_right'][1] - thresh
            x = 2 * midpoint[0] - \
                list_of_coordinate_dict['bottom_right'][0] - thresh
            list_of_coordinate_dict['top_left'] = (x, y)
        elif index == 3:  # "top_right"
            midpoint = np.add(
                list_of_coordinate_dict['top_left'], list_of_coordinate_dict['bottom_right']) / 2
            y = 2 * midpoint[1] - \
                list_of_coordinate_dict['bottom_left'][1] - thresh
            x = 2 * midpoint[0] - \
                list_of_coordinate_dict['bottom_left'][0] - thresh
            list_of_coordinate_dict['top_right'] = (x, y)
        elif index == 0:  # "bottom_left"
            midpoint = np.add(
                list_of_coordinate_dict['top_left'], list_of_coordinate_dict['bottom_right']) / 2
            y = 2 * midpoint[1] - \
                list_of_coordinate_dict['top_right'][1] - thresh
            x = 2 * midpoint[0] - \
                list_of_coordinate_dict['top_right'][0] - thresh
            list_of_coordinate_dict['bottom_left'] = (x, y)
        elif index == 1:  # "bottom_right"
            midpoint = np.add(
                list_of_coordinate_dict['bottom_left'], list_of_coordinate_dict['top_right']) / 2
            y = 2 * midpoint[1] - \
                list_of_coordinate_dict['top_left'][1] - thresh
            x = 2 * midpoint[0] - \
                list_of_coordinate_dict['top_left'][0] - thresh
            list_of_coordinate_dict['bottom_right'] = (x, y)

        return list_of_coordinate_dict


    def crop_card(self, img):
        results = self.corner_predict(img)
        final_result_array = json.loads(results)
        length_arr = len(final_result_array)
        if length_arr > 4:
            final_result_array = final_result_array[length_arr - 4:length_arr]

        dict_of_coordinates_tuple = {}

        for result in final_result_array:
            name = result['name']
            x_mean = (result['xmin'] + result['xmax']) / 2
            y_mean = (result['ymin'] + result['ymax']) / 2
            dict_of_coordinates_tuple[name] = (x_mean, y_mean)
        if len(dict_of_coordinates_tuple.keys()) == 3 or len(dict_of_coordinates_tuple.keys()) >= 4:
            if len(dict_of_coordinates_tuple.keys()) == 3:
                num_of_corner = 3
                full_coordinates = self.calculate_missed_coord_corner(
                    dict_of_coordinates_tuple)
            elif len(dict_of_coordinates_tuple.keys()) >= 4:
                num_of_corner = 4
                full_coordinates = dict_of_coordinates_tuple
            source_points = np.float32([
                list(full_coordinates['top_left']), list(
                    full_coordinates['top_right']),
                list(full_coordinates['bottom_right']), list(
                    full_coordinates['bottom_left'])
            ])
            img = np.asarray(img)

            crop = self.perspective_transform(img, source_points)
            im = Image.fromarray(crop)
            return num_of_corner, im
        else:
            print('Sorry, Something went wrong! Can you try another image!')
            pass
