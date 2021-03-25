import matplotlib
# matplotlib.rc('font', family='TakaoPGothic')
from matplotlib import pyplot as plt

import matplotlib.patches as patches
import cv2, os, csv
from viText.tools.common import poly, get_list_file_in_folder, get_list_gt_poly, type_map

color_map = {1: 'r', 15: 'green', 16: 'blue', 17: 'm', 18: 'cyan'}
txt_color_map = {1: 'b', 15: 'green', 16: 'blue', 17: 'm', 18: 'cyan'}
inv_type_map = {v: k for k, v in type_map.items()}


def viz_poly(img, list_poly, save_viz_path=None, ignor_type=[1]):
    '''
    visualize polygon
    :param img: numpy image read by opencv
    :param list_poly: list of "poly" object that describe in common.py
    :param save_viz_path:
    :return:
    '''
    fig, ax = plt.subplots(1)
    fig.set_size_inches(20, 20)
    plt.imshow(img)

    for polygon in list_poly:
        ax.add_patch(
            patches.Polygon(polygon.list_pts, linewidth=2, edgecolor=color_map[polygon.type], facecolor='none'))
        draw_value = polygon.value
        if polygon.type in ignor_type:
            draw_value = ''
        plt.text(polygon.list_pts[0][0], polygon.list_pts[0][1], draw_value, fontsize=20,
                 fontdict={"color": txt_color_map[polygon.type]})
    # plt.show()

    if save_viz_path is not None:
        print('Save visualized result to', save_viz_path)
        fig.savefig(save_viz_path, bbox_inches='tight')


def viz_icdar(img_path, anno_path, save_viz_path=None, extract_kie_type=False, ignor_type=[1]):
    if not isinstance(img_path, str):
        image = img_path
    else:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    list_poly = []
    with open(anno_path, 'r', encoding='utf-8') as f:
        anno_txt = f.readlines()

    for anno in anno_txt:
        anno = anno.rstrip('\n')

        idx = -1
        for i in range(0, 8):
            idx = anno.find(',', idx + 1)

        coordinates = anno[:idx]
        val = anno[idx + 1:]
        type = 1
        if extract_kie_type:
            last_comma_idx = val.rfind(',')
            type_str = val[last_comma_idx + 1:]
            val = val[:last_comma_idx]
            if type_str in inv_type_map.keys():
                type = inv_type_map[type_str]

        coors = [int(f) for f in coordinates.split(',')]
        pol = poly(coors, type=type, value=val)
        list_poly.append(pol)
    viz_poly(img=image,
             list_poly=list_poly,
             save_viz_path=save_viz_path,
             ignor_type=ignor_type)


def viz_icdar_multi(img_dir, anno_dir, save_viz_dir, extract_kie_type=False, ignor_type=[1]):
    list_files = get_list_file_in_folder(img_dir)
    for idx, file in enumerate(list_files):
        if idx < 0:
            continue
        # if 'mcocr_public_145014smasw' not in file:
        #     continue
        print(idx, file)
        img_path = os.path.join(img_dir, file)
        anno_path = os.path.join(anno_dir, file.replace('.jpg', '.txt'))
        save_img_path = os.path.join(save_viz_dir, file)
        viz_icdar(img_path, anno_path, save_img_path, extract_kie_type, ignor_type)



if __name__ == '__main__':
    img_path = '/data20.04/data/MC_OCR/mcocr2021_public_train_test_data/mcocr_public_train_test_shared_data/mcocr_val_data/val_images/mcocr_val_145114budzl.jpg'
    anno_path = '/home/cuongnd/PycharmProjects/aicr/PaddleOCR/inference_results/server_DB/output_txt/mcocr_val_145114budzl.txt'
    save_vix_path = 'test.jpg'
    # viz_icdar(img_path=img_path,
    #           anno_path=anno_path,
    #           save_viz_path=save_vix_path)

    viz_icdar_multi(
        '/data20.04/data/data_Korea/WER_20210122/jpg',
        '/data20.04/data/data_Korea/WER_20210122/anno_icdar',
        '/data20.04/data/data_Korea/WER_20210122/viz_anno',
        ignor_type=[],
        extract_kie_type=False)


    # csv_file = '/data20.04/data/MC_OCR/output_results/EDA/mcocr_train_df_filtered_rotate_new.csv'
    # img_dir = '/data20.04/data/MC_OCR/output_results/key_info_extraction/train_combine_lines/refine/imgs'
    # viz_dir='/data20.04/data/MC_OCR/output_results/EDA/train_visualize_filtered_rotate'
    # viz_csv(csv_file=csv_file,
    #           viz_dir=viz_dir,
    #           img_dir=img_dir)

    # img_dir = '/data20.04/data/MC_OCR/output_results/key_info_extraction/private_test_pick/imgs'
    # output_txt_dir = '/data20.04/data/MC_OCR/output_results/key_info_extraction/private_test_pick/output/txt'
    # output_viz_dir =  '/data20.04/data/MC_OCR/output_results/key_info_extraction/private_test_pick/output/viz_imgs_new'
    #