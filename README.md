# Interpretable artificial intelligence-based app assists inexperienced radiologists in diagnosing biliary atresia from sonographic gallbladder images

![https://opensource.org/licenses/MIT](https://img.shields.io/badge/license-MIT-green.svg)
![https://www.python.org/](https://img.shields.io/badge/language-python-yellow.svg)
![https://pytorch.org/get-started/locally/](https://img.shields.io/badge/backbone-PyTorch-important.svg)

# Content
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Start to run](#start-to-run)
- [Result](#result)
- [CAM](#cam)
- [License](#license)

# Overview
In this study, a new model was retrospectively trained using 3659 original sonographic gallbladder images and their derived 51226 smartphone photos taken by 7 radiologists, and tested on 11410 external validation smartphone photos.
The new smartphone app model showed robust and satisfactory performance for the diagnosis of BA based on the smartphone photos of gallbladder images, and it could aid both junior and senior radiologists to improve their diagnostic performances in real-world settings.

# System Requirements
## Hardware requirements
The source code require at lest 6GB GPU memory to support it to work.

## Software requirements
### OS Requirements
This package is supported for *Windows* and *Linux*. The package has been tested on the following systems:
+ Windows: Microsoft Windows 10 Pro
+ Linux: Ubuntu 18.04

### Python Dependencies
See requirements.txt:



# Start to run
The defult hyperparameter was set on `./src/tool.py`, and should first configure the **DataSet Path** on `./src/config.py`, and configure the **CSV file of Label** 
on `/script/run.sh`. You can enter `bash run.sh` in the terminal to run the code.

Some examples are on the `./example_data/`.


# Result
The new model significantly outperformed the previous published model in diagnosing BA on the external validation set (AUC 0.924 vs 0.908, P=0.004) with higher consistency (average kappa value 0.767 vs 0.671).

# CAM
In our study, we used `./src/util/HeatMap.py`  **the class activation map (CAM)** technology to explain on which region(s) the model focus.

# License
This project is covered under the **MIT** License.

