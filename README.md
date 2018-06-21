# Setting up the machine + virtual environment

## Create a new server
These instructions are for Paperspace. You could use AWS as well, but [Paperspace offers better value for money](https://medium.com/initialized-capital/benchmarking-tensorflow-performance-and-cost-across-different-gpu-options-69bd85fe5d58) at this point in time.

1. Sign up and create a P4000 machine with [ML-in-Box template](https://paperspace.zendesk.com/hc/en-us/articles/115002305973-Machine-Learning-in-a-Box) on [Paperspace](www.paperspace.com)
2. SSH and change password with `$ passwd` 
3. You can double-check CUDA, cuDNN and Nvidia driver versions with:
`$ (cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2 && nvcc --version && nvidia-smi)`
or 
`$ (cat /usr/include/cudnn.h | grep CUDNN_MAJOR -A 2 && nvcc --version && nvidia-smi)`
4. Move dotfiles (.bashrc, .tmuxconf, .dircolors, etc.) to set up initial configuration
5. Install tmux with `$ sudo apt install tmux`
6. Re-run bashrc with `$ source ~/.bashrc` (the `conda activate` commands will fail in the tmux windows - don't need to worry about that)


## Build a virtual environment
I personally prefer virtualenv (and virtualenvwrapper) but since the machine comes with anaconda, we'll go with that.

1. Update conda with `conda update -n base conda`
2. Create new anaconda environment with `conda create -n tensorflow_py36 python=3.6 pip`
3. Activate new environment with `source activate tensorflow_p36`
4. Install tensorflow from pre-built binaries that come with the machine, found in the [`src` folder](https://paperspace.zendesk.com/hc/en-us/articles/115002305973-Machine-Learning-in-a-Box) `pip install src/tensorflow-1.7.0-cp36-cp36m-linux_x86_64.whl`
5. Install keras with `pip install keras`
6. Reboot with `$ sudo reboot`
7. At this point, test that tensorflow is working on GPU as expected
    * `$ python`
    * `>>> import tensorflow as tf`
    * `>>> sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))`
8. Finally, create a new kernel for Jupyter notebooks inside the virtual env with `$ ipython kernel install --user --name=tensorflow_p36`

## Clone repo & set up the training, validation and test data 
1. Install aws cli with `$ pip install awscli` and configure with `$ aws configure`
2. Create directories `$ mkdir WIP && mkdir WIP/180503_lentil_app && mkdir WIP/180503_lentil_app/imgs cd WIP/180503_lentil_app/`
3. Clone git repo `$ git clone https://github.com/DeepBodapati/lentil_app.git .`
4. Copy all the S3 files 
    `$ aws s3 cp s3://lentil-imgs/src_imgs_from_eBay.zip imgs/ && aws s3 cp s3://lentil-imgs/src_imgs_from_home.zip imgs/ && aws s3 cp s3://lentil-imgs/test_imgs_from_DeeAnna.zip imgs/`
5. Unzip all the downloaded S3 files:
    * `$ cd imgs/`
    * `$ unzip src_imgs_from_eBay.zip && mv imgs/ src/`
    * `$ unzip src_imgs_from_home.zip && rsync -a imgs_from_home/ src/ && rm -rf imgs_from_home`
    * `$ unzip test_imgs_from_DeeAnna.zip -d test/`
6. Run the `prep_data_for_DL-ebay-only.ipynb` notebook to separate into training and validation data

## Train the model 
1. Run `Xception_fine_tuning.ipynb` notebook to train model
