# OccRob

### Dependency

**Install Anaconda**

We use Anaconda to manage python environment, after anaconda installed, create a virtual environment with python3.7.

```shell
conda create -n occrob python=3.7
conda activate occrob
```

**Python Package**

```shell
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cpuonly -c pytorch
# other necessary packages
onnx=1.10.2
onnxruntime=1.10.0
pandas=1.3.4
matplotlib=3.2.2
pebble=4.6.3
```

**Install Marabou**

You can download Marabou from https://github.com/NeuralNetworkVerification/Marabou and install Marabou following the instructions.
Note we use maraboupy and Gurobi with Marabou.

**Install Gurobi**

Installation of Gurobi needs network environment during the whole process. After installation, a license is needed to use Gurobi and the instructions can be fonud at Gurobi's web page.

### Run OccRob

You can download the model from Dropbox with the following code and unzip it to the root of the project.

```shell
wget https://www.dropbox.com/s/mesvva28fvc3rla/model.zip?dl=1
sudo unzip model.zip
```

Then configure the project root into environment variable.

```shell
export PYTHONPATH=$PYTHONPATH:/path/to/project/root
```

Then you would be able run occrob in `mnist/experiment` and `gtsrb/experiment` with the following code:

```shell
# run uniform occlusion experiment with default command-line parameters.
python uniform_occlusion.py
# python uniform_occlusion.py --model small --testNum 30
# run multiform occlusion experiment with default command-line parameters.
python multiform_occlsion.py
# python multiform_occlusion.py --model small --testNum 30 --epsilon 0.2
```
