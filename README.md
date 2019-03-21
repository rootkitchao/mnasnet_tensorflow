# MNASNET Tensorflow

 An unofficial implementation of MNASNet.
 Use Tensorflow Slim.
 MNASNet is a CNN model proposed by Google for mobile devices.It is faster and more accurate than previous models, such as Mobilenet V2 and Shufflenet V2.

 References: [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/pdf/1807.11626.pdf) by Mingxing Tan, et. al.

## Object Detection
  We tested the replacement of ResNet with MNASNet-b1 as the backbone of RetinaNet.Evaluation on the MSCOCO17 val after training on the MSCOCO17 train dataset.We have achieved a mAP of 33.7.We have achieved a mAP of 33.7, which is slightly better than using MobileNet V2 as the backbone(33.7 vs 33.4).All work on object detection uses the [object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection) of tensorflow.We just wrote a feature extractor and related configuration files.

## How To Use
 The basic implementation of Mnasnet is in the /mnasnet directory. If you want to use it, copy it to the /slim/nets directory, then add mnasnet to [nets_factory.py](https://github.com/tensorflow/models/blob/r1.13.0/research/slim/nets/nets_factory.py).
 
```python
networks_map = {
                #......
                'mnasnet_a1':mnasnet.mnasnet_a1,
                'mnasnet_b1':mnasnet.mnasnet_b1
               }

arg_scopes_map = {
                  #......
                  'mnasnet_a1':mnasnet.training_scope,
                  'mnasnet_b1':mnasnet.training_scope,
                 }
```
 If you want to use mnasnet in the [object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection), we provide the implementation of retinanet.Copy the files in the /object_detection/models directory to where the object detection api is located, then add the model to [model_builder.py](https://github.com/tensorflow/models/blob/r1.13.0/research/object_detection/builders/model_builder.py).The configuration file is provided in the /object_detection/samples/configs/ directory
 
```python
SSD_FEATURE_EXTRACTOR_CLASS_MAP = {
    #......
    'ssd_mnasnet_a1_fpn':SSDMNASNetA1FpnFeatureExtractor,
    'ssd_mnasnet_b1_fpn':SSDMNASNetB1FpnFeatureExtractor,
}
```
## Pretrained models

### Imagenet Model
| Model | Input Size | Depth Multiplier | Top-1 Accuracy | Top-5 Accuracy | Pixel 1 latency (ms) | DownLoad Link |
| :---- | ---------- | ---------------- | -------------- | -------------- | -------------------- | ------------- |
| mnasnet-b1 | 224*224 | 1.0 | 74.094 | 92.002 | TBA | [mnasnet_b1_1.0_224.tar](https://drive.google.com/open?id=1A04CaDk6WhXCwZ1ivkLQxE1YhPV1WYcz)
| mnasnet-a1 | 224*224 | 1.0 | TBA | TBA | TBA | TBA

### MSCOCO17 Model
| Model | Input Size | Depth Multiplier | mAP(IOU 0.05:0.95) | mAP.L | mAP.M | mAP.S |  Pixel 1 latency (ms) | DownLoad Link |
| :---- | ---------- | ---------------- | ------------------ | ----- | ----- | ----- | -------------- | -------------------- |
| ssd-mnasnet-b1-fpn | 640*640 | 1.0 | 33.7 | 47.7 | 30.5 | 11.7 | TBA | [ssd_mnasnet_b1_fpn_shared_box_predictor_640x640_coco17_2019_03_21.tar](https://drive.google.com/open?id=1t6WAYdG5lYMd2pOKS89hTE-UOpy4KrEh)
| ssd-mobilenet-v1-fpn | 640*640 | 1.0 | 32.5 | 46.0 | 29.2 | 10.1 | TBA | [ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco17_2019_03_21.tar](https://drive.google.com/open?id=1LtvJqFGuALjBiDKkyLY2xDM_UGw1NAFD)
| ssd-mobilenet-v2-fpn | 640*640 | 1.0 | 33.4 | 47.2 | 30.5 | 10.7 | TBA | [ssd_mobilenet_v2_fpn_shared_box_predictor_640x640_coco17_2019_03_21.tar](https://drive.google.com/open?id=1VqpJ_DAZtmjrM8oAkCDP1dLIJRnHzY-Y)

## System Requirement

  Tensorflow 1.X  

## TODO List
 
  * [x] Upload the code for object detection
  * [x] Freeze the object detection model and upload it 
  * [ ] Training the MNASNet A1 model (in progress, estimated to take 60 days)
  * [ ] Evaluation on mobile devices (I am sorry that I don't have a Google Pixel 1 phone, I might buy a newer phone for evaluation)
  * [ ] Provide more pre-trained models (our limited computing power)
  * [ ] Send PR to tensorflow

## License
 
 [Apache License 2.0](LICENSE)