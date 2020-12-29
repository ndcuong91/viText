from PIL import Image
import cv2

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

debug = False
eval = True


class Classifier_Vietocr:
    def __init__(self, ckpt_path=None,
                 gpu='0',
                 config_name='vgg_seq2seq'):
        print('Classifier_Vietocr. Init')
        self.config = Cfg.load_config_from_name(config_name)

        # config['weights'] = './weights/transformerocr.pth'
        if ckpt_path is not None:
            self.config['weights'] = ckpt_path
        self.config['cnn']['pretrained'] = False
        if gpu is not None:
            self.config['device'] = 'cuda:' + str(gpu)
        self.config['predictor']['beamsearch'] = False
        self.model = Predictor(self.config)

    def inference(self, numpy_list, debug=False):
        print('Classifier_Vietocr. Inference',len(numpy_list),'boxes')
        text_values = []
        for idx, f in enumerate(numpy_list):
            img = Image.fromarray(f)
            s = self.model.predict(img)
            if debug:
                print(s)
                cv2.imshow('sample',f)
                cv2.waitKey(0)
            text_values.append(s)
        return text_values


def test_inference():
    engine = Classifier_Vietocr(ckpt_path='weights/vgg19_bn_seq2seq.pth')

    img_path = 'sample.jpeg'

    numpy_list=[cv2.imread(img_path)]
    a, b = engine.inference(numpy_list, debug=True)


if __name__ == "__main__":
    test_inference()
