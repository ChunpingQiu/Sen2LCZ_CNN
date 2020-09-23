# LCZ classification pytorch
A simple example for LCZ classification using pytorch, based on template [here](https://github.com/victoresque/pytorch-template).

## Content and Folder Structure

### LCZ related

- **LCZDataLoader** in ```data_loader/data_loaders.py```
- **LCZ model** Net_LCZ in ```model/model.py```
- training and validation setting in ```config.json```
- basic test about the dataloader and model ```baselineNN_notebooks```

### basics from the template
```
pytorch-template/
│
├── train.py - main script to start training
├── test.py - evaluation of trained model
│
├── config.json - holds configuration for training
├── parse_config.py - class to handle config file and cli options
│
│
├── base/ - abstract base classes
│   ├── base_data_loader.py
│   ├── base_model.py
│   └── base_trainer.py
│
├── data_loader/ - anything about data loading goes here
│   └── data_loaders.py
│
├── data/ - default directory for storing input data
│
├── model/ - models, losses, and metrics
│   ├── model.py
│   ├── metric.py
│   └── loss.py
│
├── saved/
│   ├── models/ - trained models are saved here
│   └── log/ - default logdir for tensorboard and logging output
│
├── trainer/ - trainers
│   └── trainer.py
│
├── logger/ - module for tensorboard visualization and logging
│   ├── visualization.py
│   ├── logger.py
│   └── logger_config.json
│  
└── utils/ - small utility functions
    ├── util.py
    └── ...
```
## Usage

Modify the configurations in `.json` config files, then run:

  ```
  python train.py --config config.json
  ```


<!-- [//]: # (- predictions from tif files:)

[//]: # (### Investigations on LCZ classification task)

### Investigations on MTL
- define/setup models in modelS.py `CUDA_VISIBLE_DEVICES=0 python plotModel.py`
- `CUDA_VISIBLE_DEVICES=N python train.py --methods4test w_learned --folderData './mtl_SampleData/patches/' --saveFolder './results/'`
- predictions from tif files: `CUDA_VISIBLE_DEVICES=0 python img2map.py --methods4test w_learned --modelPath './results/' --tifFile './mtl_SampleData_tif/henan_2017_sentinel_22.tif' --modelWeights "weights.best_lcz"` -->

## datasets
- some sample data:


## td list
- [ ] test on the trained model
- [ ] visualize the training process
- [ ] using gpu for model training
- [x] dataloder test and training

<!---
[//]: # (- [x] predict with the trained model)
- [x] test different models with the same data
- [x] training different models under the same configuration
- [x] check created patches
- [x] from images to patches
-->
