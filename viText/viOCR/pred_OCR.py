from viText.viOCR.sources.crnn.crnn_class import Classifier_CRNN
from viText.viOCR.sources.vietocr.vietocr_class import Classifier_Vietocr
import cv2

img_path = 'data/sample.jpeg'


def pred_crnn():
    img_data = cv2.imread(img_path)
    ckpt_path = 'weights/crnn_vgg.pth'
    classifier = Classifier_CRNN(ckpt_path=ckpt_path,
                                 imgW=512,
                                 imgH=64,
                                 gpu='0')
    values, probs = classifier.inference([img_data])
    print(values, probs)


def pred_vietocr():
    classifier = Classifier_Vietocr(ckpt_path='weights/vgg19_bn_seq2seq.pth')

    numpy_list=[cv2.imread(img_path)]
    values = classifier.inference(numpy_list, debug=False)
    print(values)

if __name__ == "__main__":
    pred_crnn()
    pred_vietocr()