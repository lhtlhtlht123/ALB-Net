
# ALB-Net
## Prerequisites



Download our repo:
```
git clone https://github.com/lhtlhtlht123/ALB-Net.git
cd ALB-Net
```
Install packages from requirements.txt
```
pip install -r requirements.txt
```

## Datasets processing
Choose a path to create a folder with the dataset name and download datasets [DRIVE](https://www.dropbox.com/sh/z4hbbzqai0ilqht/AAARqnQhjq3wQcSVFNR__6xNa?dl=0),[CHASEDB1](https://blogs.kingston.ac.uk/retinal/chasedb1/), and [CrackTree200]([https://gitcode.com/open-source-toolkit/38f45?source_module=search_project). Type this in terminal to run the data_process_train.py file

```
python data_process_train.py
```

## Training
Type this in terminal to run the train.py file

```
python train.py -dp DATASET_PATH
```
## Test
Type this in terminal to run the test.py file

```
python test.py -dp DATASET_PATH -wp WEIGHT_FILE_PATH
```

