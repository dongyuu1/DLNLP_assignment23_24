# APPLIED MACHINE LEARNING SYSTEM II ELEC0135 

The code is for the assignment of Deep Learning for Natural Language Processing and is implemented by 
Dongyu Wang (SN:23104424). 

The code is an implementation the of Aspect-based Sentiment Analysis (ABSA). It takes advantage of DeBERTa and 
combines it with an adversarial training approach to enhance ABSA's performance. 


## Code Structure and Description
main.py is used to launch the code. ./Datasets is the folder containing the data for training and testing. 
./A contains the code dealing with the ABSA. 

In folder A, asc_launch.py contains all the code for launching the model training and evaluation. 
cfgs.py stores all the hyperparameters and config information. ./dataset includes the code related to retrieving and preprocessing the data. 
./deberta contains the code of the model architecture, which is provided by https://github.com/huberemanuel/DeBERTa/. 
./utils includes some functional code to launch the deberta model https://github.com/huberemanuel/DeBERTa/.
## Initialization

Please launch anaconda and create the environment with the following command:
```
cd path/to/DLNLP
conda env create -f environment.yaml
conda activate DLNLP
```

Please run the following command in the environment to install pytorch:
```
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

## Launch
Here we can start training of the model:
```
python main.py 
```

To adjust hyperparameters, please use the following command:
```
python main.py --lr 1e-5 --epsilon 1 --lambda_ 10
```

