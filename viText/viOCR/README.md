# crnn
This repo implemented convolution recurrent neural nets(CRNN) for handwritten recognition, optical character recognition. CRNN can use for many text levels: character, word, or even a text line.


# I. Dataset
You must organize your dataset in the below structure
```
data
├── train
│   ├── a.jpg
│   ├── a.txt
│   ├── b.jpg
│   └── b.txt
├── test
│   ├── c.jpg
│   └── c.txt
├── val.txt
└── train.txt
```
example of **train.txt**
```
train/a.jpg
train/b.jpg
```
example of **val.txt**
```
test/c.jpg
```

# II. Training
## Configs
There are some important options which you need to modify in **config_crnn.py**
* imgW: should be larger than the maximum width of images in training set.
* imgH: 64
* output_dir: where output's checkpoints store
* gpu_train: 1, 0 or None
* train_file: should be *train.txt* in Step 3 of section I. Dataset
* val_file: should be *val.txt* in Step 3 of section I. Dataset
* ckpt_prefix: should be [date of training] + [type of training] (e.g. '20200826_general')

## Train
Training process was simplified, just type the command in your terminal and you should use the pretrain model

```
python train.py
```
# III. Predict
## Configs
* imgW: should be larger than the maximum width of images in training set.
* imgH: 64
* pretrained_test: path to checkpoints
* gpu_test: 1, 0 or None
* write_predict_file: write output result into files for evaluation
* include_conf: include confident scores in predict files

# IV. Benchmark
**CER (Character error rate)**

| **engine** |  **FUNSD** | 
| -------------------- | --------- | 
| CRNN (VGG_like - BiLSTM - CTC loss)   | 0.2072  
| Vietocr (VGG19-bn - Transformer)| 0.126 | 
| Vietocr (VGG19-bn - seq2seq)| 0.1223 |


**Speed (GTX 1080Ti)**

| **engine** |  **inference time** |
| ---------- | --------- |
| CRNN (VGG_like - BiLSTM - CTC loss)   | 6ms   |
| Vietocr (VGG19-bn - Transformer)  | 86ms |
| Vietocr (VGG19-bn - seq2seq)  | 12ms |

## Sample code 
**We're still finding the best checkpoint, so this checkpoint is not final**
```
    import cv2
    from viText.viOCR.source.crnncrnn_class import Classifier_CRNN

    img_data = cv2.imread('data/sample.jpg')
    classifier = Classifier_CRNN(ckpt_path='checkpoints/20200912_ocr_dataset_64_VGG_like_32_cer_0.0257.pth', imgW=512, imgH=64, gpu='0')
    values, probs = classifier.inference([img_data])
    print(values, probs)
```
Outputs

```
    Using VGG_like backbone
    Classifier_CRNN. Use GPU 0
    Classifier_CRNN. Load checkpoint checkpoints/20200912_ocr_dataset_64_VGG_like_32_cer_0.0257.pth
    Classifier_CRNN. New W 161
    Classifier_CRNN. Begin classify 1 boxes
    Classifier_CRNN. Processing time: 0.07833743095397949
    Classifier_CRNN. Speed: 12.765289693856122 fps
    ['Hạnh'] [[0.9997, 0.9998, 0.9999, 1.0]]
```


# Dependences
* [warp_ctc_pytorch](https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding)
