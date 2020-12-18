from viText.viOCR.source.crnn.crnn_class import Classifier_CRNN
import cv2

img_path = 'crnn/data/sample.jpg'


def pred_crnn():
    img_data = cv2.imread(img_path)
    ckpt_path = '/home/cuongnd/PycharmProjects/aicr/aicr.core2/engine/classifier_crnn/classifier_crnn/' \
                'checkpoints/20200912_ocr_dataset_64_VGG_like_32_cer_0.0257.pth'
    classifier = Classifier_CRNN(ckpt_path=ckpt_path,
                                 imgW=512,
                                 imgH=64,
                                 gpu='0')
    values, probs = classifier.inference([img_data])
    print(values, probs)


if __name__ == "__main__":
    pred_crnn()