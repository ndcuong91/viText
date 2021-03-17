from viText.tools.utils import get_list_file_in_folder
import json, os, cv2


def split_receipt_from_raw_img(img_dir, anno_dir):
    list_img = get_list_file_in_folder(img_dir, ext=['jpg', 'png', 'JPG', 'PNG', 'tiff', 'TIFF'])
    for img_name in list_img:
        img = cv2.imread(os.path.join(img_dir, img_name))
        anno_path = os.path.join(anno_dir, img_name.split('.')[0] + '.json')

        with open('data.txt') as json_file:
            data = json.load(json_file)
            for bbox in data['shapes']:
                bbox_ = bbox["points"]
                kk=1


if __name__ == "__main__":
    img_dir = ''
    split_receipt_from_raw_img(img_dir=img_dir, anno_dir=img_dir)
