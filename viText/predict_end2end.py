import cv2, os, time, sys
from datetime import datetime
from detector_db.db_class import Detector_DB
from classifier_crnn.prepare_crnn_data import get_list_file_in_folder
from classifier_crnn.crnn_class import Classifier_CRNN
from classifier_crnn.onnxruntimeClass import Classifier_CRNN_onnx
from classifier_crnn.crnn_class import get_boxes_data, visualize_results
import aicr_configs.aicr.config as aicrconfig
from aicr_metric.F1_score import eval_F1, writer
from aicr_metric.Check_Performance_WER import eval_F1_WER
import shutil
from vietocr.vietocr_class import Classifier_Vietocr

os.environ["PYTHONIOENCODING"] = "utf-8"
pred_time = datetime.today().strftime('%Y-%m-%d_%H-%M')

gpu = '0'
dataset_name = 'invoice_retail'  # funsd, korea_English_test, Eval_Vietnamese or Cello_Vietnamese
eval=True
if dataset_name == 'funsd':
    img_dir = '/data20.04/data/aicr/funsd_extra/dataset/testing_data/images'
    gt_dir = '/data20.04/data/aicr/funsd_extra/dataset/testing_data/anno_voc'
elif dataset_name == 'Eval_Vietnamese':
    img_dir = '/data20.04/data/data_Korea/Eval_Vietnamese/images'
    gt_dir = '/data20.04/data/data_Korea/Eval_Vietnamese/GT_word_voc_refined1508'
elif dataset_name == 'Cello_Vietnamese':
    img_dir = '/data20.04/data/data_Korea/Cello_Vietnamese/images'
    gt_dir = '/data20.04/data/data_Korea/Cello_Vietnamese/GT_word_voc_1908'
elif dataset_name == 'korea_English_test':
    img_dir = '/data20.04/data/data_Korea/korea_English_test/images'
    gt_dir = '/data20.04/data/data_Korea/korea_English_test/GT_word_voc'
else:
    eval=False
    if dataset_name=='korea_English_test':
        img_dir='/data20.04/data/data_Korea/Korea_test_Vietnamese_1106'
    if dataset_name=='invoice_retail':
        img_dir='/data20.04/data/data_invoice/invoice_retail'

img_dir='/data20.04/data/aicr/Cello_data1'
img_path = '/data20.04/data/aicr/619327481172390.jpg'
img_path = ''
# detector
detector_ckpt_path = '/home/cuongnd/PycharmProjects/aicr/aicr.core2/engine/detector_db/detector_db/train/workspace/outputs/train_2020-12-14_19-05/model/model_epoch_945_minibatch_174000'
detector_ckpt_path = '/home/cuongnd/PycharmProjects/aicr/aicr.core2/engine/detector_db/detector_db/' \
                     'test/checkpoints/20200825_resnet50_1152_final.pth'
# detector_ckpt_path = '/data20.04/data/aicr_checkpoints/detector_db/training/train_2020-08-25_21-26_resnet50_1152/model/20200825_resnet50_1152_final'
output_dir='output/predict_end2end/others'
if img_path=='':
    output_dir = os.path.join('outputs', 'predict_end2end', dataset_name + '_' + pred_time)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
detector_box_thres = 0.4
polygon = False
detector_write_file = False
detector_visualize = True
img_short_side = 1152  # 736
deform_conv = False
resnet = '50'
parallelize = False

vietocr=False
# get bbox
crop_method = 2  # overall method 2 is better than method 1 FOR CLASSIFIER (NOT DETECTOR)
# classifier
backbone = 'VGG_like'  # VGG_like | resnet
classifier_ckpt_path = aicrconfig.crnn_checkpoint
classifier_ckpt_path = '/data20.04/data/aicr_checkpoints/classifier_crnn/training/train_2020-09-15_23-00_general_vnmese/' \
                       '20200915_general_vnmese_words_VGG_like_42_loss_0.06_cer_0.0029.pth'
classifier_ckpt_path = '/data20.04/data/aicr_checkpoints/classifier_crnn/training/' \
                       'train_2020-09-12_10-59_ocr_dataset_VGG_like/20200912_ocr_dataset_64_VGG_like_32_loss_3.06_cer_0.0257.pth'
classifier_batch_sz = 4
worker = 1
write_file = True
visualize = True
extend_bbox = True  # extend bbox when crop or not
debug = False
if gpu is None or debug:
    classifier_batch_sz = 1
    worker = 0

def main():
    begin_init = time.time()
    detector, classifier = init_models(gpu=gpu)
    end_init = time.time()
    print('Init models time:', end_init - begin_init, 'seconds')
    begin = time.time()
    list_img_path = []
    if img_path != '':
        list_img_path.append(img_path)
    else:
        list_img_path = get_list_file_in_folder(img_dir)
    list_img_path = sorted(list_img_path)
    for img in list_img_path:
        print('\nInference', img)
        test_img = cv2.imread(os.path.join(img_dir, img))
        begin_detector = time.time()
        boxes_list = detector.inference(os.path.join(img_dir, img), debug_write_outputs=detector_write_file,
                                        debug_visualize=detector_visualize)
        end_detector = time.time()
        print('Detector time:', end_detector - begin_detector, 'seconds')

        boxes_data, boxes_info = get_boxes_data(test_img, boxes_list, extend=False, crop_method=crop_method)
        boxes_data2, boxes_info = get_boxes_data(test_img, boxes_list, extend=extend_bbox, crop_method=crop_method)
        end_get_boxes_data = time.time()
        # print('Get boxes time:', end_get_boxes_data - end_detector, 'seconds')
        #
        values, probs = classifier.inference(boxes_data2)

        for idx, box in enumerate(boxes_info):
            box.asign_value(values[idx])

        end_classifier = time.time()
        print('Classifier time:', end_classifier - end_get_boxes_data, 'seconds')
        print('Predict time:', end_classifier - begin_detector, 'seconds')
        if write_file:
            write_output(boxes_info, os.path.join(output_dir, os.path.basename(img).split('.')[0] + '.txt'))

        if visualize:
            try:
                visualize_results(test_img, os.path.basename(img), boxes_info, output_dir=output_dir)
            except:
                print('visualize error')
            end_visualize = time.time()
            print('Visualize time:', end_visualize - end_classifier, 'seconds')

    end = time.time()
    speed = (end - begin) / len(list_img_path)
    print('Processing time:', end - begin, 'seconds. Speed:', round(speed, 4), 'second/image')
    print('Done')


def init_models(gpu='0'):
    if gpu != None:
        print('Use GPU', gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    else:
        print('Use CPU')

    detector = Detector_DB(ckpt_path=detector_ckpt_path, gpu=gpu, image_short_side=img_short_side,
                           polygon=polygon, output_dir=output_dir,
                           box_thres=detector_box_thres, parallelize=parallelize,
                           deform_conv=deform_conv, resnet=resnet)
    if vietocr:
        classifier = Classifier_Vietocr()
    else:
        classifier = Classifier_CRNN(ckpt_path=classifier_ckpt_path, backbone=backbone,
                                 batch_sz=classifier_batch_sz, gpu=gpu, workers=worker)


    # classifier = Classifier_CRNN_onnx(ckpt_path=aicrconfig.crnn_checkpoint_onnx,batch_sz=classifier_batch_sz,)
    return detector, classifier


def write_output(boxes_info, result_file_path, format='DB'):  # format = 'DB' or 'VOC'
    result = ''
    for box in boxes_info:
        top = str(box.ymin)
        left = str(box.xmin)
        bottom = str(box.ymin + box.height)
        right = str(box.xmin + box.width)
        line = ','.join([left, top, right, top, right, bottom, left, bottom, box.value])
        result += line + '\n'
    with open(result_file_path, 'w', encoding='utf8') as res:
        res.write(result)



if __name__ == '__main__':
    main()
    saved = sys.stdout
    log_file = os.path.join(output_dir, 'evaluation_result.txt')
    f = open(log_file, 'w')
    sys.stdout = writer(sys.stdout, f)
    if eval:
        print('\nStart evaluation...')
        print('Detector ckpt', detector_ckpt_path)
        print('Detector box_thres', detector_box_thres)
        print('Detector img_short_side', img_short_side)
        print('Clasfifier_ckpt', classifier_ckpt_path)
        print('extend bbox', extend_bbox)
        print()
        f1, wer = eval_F1_WER(path_img=img_dir, path_gt_box=gt_dir, path_pr_box=output_dir)

        shutil.move(output_dir, output_dir + '_' + str(f1) + '_' + str(wer))
        # eval_F1(gt_dir=gt_dir, pred_dir=output_dir)
    sys.stdout = saved
    f.close()
