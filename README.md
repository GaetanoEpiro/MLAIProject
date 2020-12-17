# JiGen AIMLProject  <img src="https://github.com/silvia1993/Jigen_AIMLProject/blob/main/aiml.png" align="right" width="200">

Basic code to reproduce DG and DA baselines related to Table 1 and Table 4 of [Jigen official paper](https://arxiv.org/pdf/1903.06864.pdf).

## Dataset

1 - Download PACS dataset from here http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017

2 - Place the dataset in the Jigen_AIMLProject folder making sure that the images are organized in this way:

```
PACS/kfold/art_painting/dog/pic_001.jpg
PACS/kfold/art_painting/dog/pic_002.jpg
PACS/kfold/art_painting/dog/pic_003.jpg
...
```

## Pretrained model

For the DG experiments, in order to reproduce exactly the values reported in the tables, you have to use the "caffe" pretrained model. 

You can download it from here 

https://drive.google.com/file/d/1wUJTH1Joq2KAgrUDeKJghP1Wf7Q9w4z-/view?usp=sharing 

Then, you have to put it into 

```
/Jigen_AIMLProject/models/pretrained/
```

## Environment

To run the code you have to install all the required libraries listed in the "requirements.txt" file.

For example, if you read

```
torch==1.4.0
```

you have to execute the command:

```
pip install torch==1.4.0

```

## Experiments

Once all is set up, you can launch the experiments listed in the "train.sh" file making sure to modify the "--path_dataset" flag with your personal path.

I recommend you to run at least 3 runs for each experiment in order to have more accurate values.
