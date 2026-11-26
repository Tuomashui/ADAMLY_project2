# IMDb and BANKING77 classification

The project has two scripts:

- Code_Week_1.py runs the original IMDb baseline and exploration.
- Code_Week_2.py runs the Week 5 baseline, Transformer, and BERT experiments.

All plots are saved as PNG images in the single figures folder.
Scores and text-length statistics are printed in the terminal.
Every run trains the selected models again.

## Install

Use your regular Python installation. In PowerShell:

~~~powershell
cd C:\Users\Admin\Documents\GitHub\ADAMLY_project2
$projectPython = "$env:LOCALAPPDATA\Python\pythoncore-3.15-64\python.exe"
& $projectPython -m pip install --upgrade -r requirements.txt
~~~

The existing Python 3.15 installation already has these dependencies.

## Run Week 5

~~~powershell
# Run the complete assignment.
& $projectPython Code_Week_2.py

# Run one part.
& $projectPython Code_Week_2.py --part imdb
& $projectPython Code_Week_2.py --part banking77
& $projectPython Code_Week_2.py --part transfer

# Small real-data check of every part, using one epoch.
& $projectPython Code_Week_2.py --quick

# Change the maximum number of training epochs.
& $projectPython Code_Week_2.py --epochs 10
~~~

The default uses five epochs, seed 42, batch size 64, and a maximum of 128
tokens per text. Settings are near the top of Code_Week_2.py.
The default pretrained model is prajjwal1/bert-tiny.
The layer study uses BANKING77 and compares head-only, upper-block,
lower-block, and full fine-tuning. Each starts from the same pretrained
checkpoint and random seed.

Training and validation data use the same stratified split for every model.
The best neural checkpoint is selected in memory using validation accuracy
before test evaluation. Training stops after two epochs without improvement.
Comparison figures show test accuracy and macro-F1.
Pareto plots show accuracy versus trainable parameter count.

Quick-run figures have a quick_ prefix. Normal runs overwrite their figure
names. Each run prints its results and saves PNG figures, including a score table.

Hugging Face maintains its download cache. Existing downloads in the hidden
.cache folder are reused when available.

## Run Week 1

~~~powershell
& $projectPython Code_Week_1.py
~~~

Week 1 saves its plots in the same figures folder using a week1_ prefix.
Word clouds are optional. If wordcloud is unavailable, the script still
produces word-frequency plots and trains the classifier.
