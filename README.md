Reinforcing Knowledge Distillation for Stance Detection

![https://github.com/PQH-hub/stance_detection/blob/main/%E5%9B%BE%E7%89%871.png]

**This is the data and code for our paper** Reinforcing Knowledge Distillation for Stance Detection.



## Prerequisites

Make sure your local environment has the following installed:

```
numpy==1.22.4
pandas==1.3.5
scikit-learn==1.2.2
scipy==1.9.3
torch==1.13.0
transformers==4.15.0
wordninja==2.0.0
tweet-preprocessor==0.6.0
```

## Datasets

We provide the dataset in the [data](https://github.com/Cheng0829/Fuzzy-DDI/blob/master/data) folder.

| Data         | Description                                                  |
| ------------ | ------------------------------------------------------------ |
| **COVID-19** | It is a dataset collected during the COVID-19 pandemic for stance detection. It contains four themes: " Face Mask," " Fauci," " Stay at Home," and " School Closure." The dataset has been annotated to determine whether the authors are for, neutral to, or against these topics. |
| **P-Stance** | It is a dataset collected during the 2020 U.S. presidential election for stance detection. It contains the names "Donald Trump," "Joe Biden," and "Bernie Sanders." The dataset has been annotated to determine the authors' positions for or against these presidential candidates. |

## Documentation

```
--config
	|--config-bert.txt
--data
	|--covid-19
	|--pstance
--src
	|--utils
		|--data_helper.py
		|--evaluation.py
		|--modea_calib.py
		|--model_utils.py
		|--modeling.py
		|--preprocessing.py
	|--emnlp_dict.txt
	|--noslang_data.json
	|--requirements.txt
	|--train.sh
	|--train_model.py
--vinai
	|--bertweet-base
--.gitignore
--README.md
```

## Train

Training script example:

```
./src/train_model.py
```

*TODO: More training scripts for easy training will be added soon.*

## Test

The trained model will be automatically stored under the folder 

```
./model
```

. The model name will be 

```
[Bertweet_Number_seedNumber].pt
```

*TODO: More pretrained models will be uploaded soon*

## Authors

PQH-hub@github.com

Email:z2017191617@outlook.com&2017191617@qq.com

Site: [GitHub]()

