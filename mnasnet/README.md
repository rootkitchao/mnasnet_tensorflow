# MNASNet
This code was implemented based on results in the Mnasnet paper.
Compared with Mobilenet V2, Mnasnet can achieve higher accuracy or faster speed on mobile devices.
References: [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/pdf/1807.11626.pdf) by Mingxing Tan, et. al.

## Train
To train mnasnet with slim's `train_image_classifier`, you need to start with linearly increase the learning rate from 0 to 0.256.

```bash
--model_name="mnasnet_b1"
--dataset_name=imagenet
--dataset_split_name=train
--preprocessing_name="inception_v2"
--train_image_size=224
--label_smoothing=0.1
--moving_average_decay=0.9999
--weight_decay=0.00001
--batch_size= 96
--learning_rate_decay_type=polynomial
--learning_rate=0
--end_learning_rate=0.256*NUM_GPUS
--learning_rate_decay_factor=1
--num_epochs_per_decay=1/NUM_GPUS
--num_clones = NUM_GPUS
--preprocessing_name=inception_v2
```

After that, start training with the hyperparameters in the paper.

```bash
--model_name="mnasnet_b1"
--dataset_name=imagenet
--dataset_split_name=train
--preprocessing_name="inception_v2"
--train_image_size=224
--label_smoothing=0.1
--moving_average_decay=0.9999
--weight_decay=0.00001
--batch_size= 96
--learning_rate_decay_type=exponential
--learning_rate=0.256*NUM_GPUS
--learning_rate_decay_factor=0.97
--num_epochs_per_decay=2.4/NUM_GPUS
--num_clones = NUM_GPUS
--preprocessing_name=inception_v2
```

We trained mnasnet b1 on two NVIDIA Geforce RTX2080TIs and achieved the accuracy given by the paper after training 4.2M steps.

## Pretrained models
| Model | Input Size | Depth Multiplier | Top-1 Accuracy | Top-5 Accuracy | Pixel 1 latency (ms) | DownLoad Link |
| :---- | ---------- | ---------------- | -------------- | -------------- | -------------------- | ------------- |
| mnasnet-b1| 224*224 | 1.0 | 74.094 | 92.002 | TBA | [mnasnet_b1_1.0_224.tar](https://drive.google.com/open?id=1A04CaDk6WhXCwZ1ivkLQxE1YhPV1WYcz)


