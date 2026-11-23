# IMDb and BANKING77 classification

Run commands from the repository root in PowerShell.

## Install with regular Python

The project can run directly with a regular Python installation.
Install the newest available dependencies into the Python you will use:

~~~powershell
$projectPython = "$env:LOCALAPPDATA\Python\pythoncore-3.15-64\python.exe"
& $projectPython -m pip install --upgrade -r requirements.txt
& $projectPython -m pip check
~~~

The example selects your installed Python 3.15 directly. Change
`$projectPython` to use another Python installation. Requirements have no
version pins. Install the dependencies into each Python installation you choose.

Compatibility was checked with Python 3.15.2, PyTorch 2.15.0,
Transformers 5.24.0, and datasets 5.2.0 by completing all eight
one-epoch smoke experiments on real dataset subsets.

Run the commands below from the repository root. In an editor, select the
same Python interpreter used to install the dependencies. PowerShell
activation scripts are unnecessary for these commands.

The first run downloads datasets and pretrained weights from Hugging Face.
Caches and large model artifacts are excluded from Git.

Week 1 additionally supports optional word clouds through the wordcloud package.
Install it separately with the following command when a compatible wheel or
a C++ compiler is available:

~~~powershell
& $projectPython -m pip install --upgrade wordcloud
~~~

Without this optional package, Week 1 still produces its word-frequency plots
and classification results.

## Run Week 5

~~~powershell
# Verify all stages on small real-data subsets (one epoch).
& $projectPython Code_Week_2.py --quick

# Complete full-data experiments, with transfer learning on BANKING77.
& $projectPython Code_Week_2.py

# Individual assignment parts, using the same full-run configuration.
& $projectPython Code_Week_2.py --part imdb
& $projectPython Code_Week_2.py --part banking77
& $projectPython Code_Week_2.py --part transfer
& $projectPython Code_Week_2.py --part report
~~~

The default is CPU when CUDA is unavailable. It uses four CPU threads,
a maximum of five epochs, early-stopping patience two, batch size 64,
and input length 128. The pretrained default is `prajjwal1/bert-tiny`.
The scratch Transformer starts with random weights.

Use `--dataset both` to run the layer study on both datasets.
Use `--checkpoint bert-base-uncased` for a larger BERT model.
Choose a fresh `--output` directory when changing either option.
Changed settings require a fresh `--output` directory:

~~~powershell
& $projectPython Code_Week_2.py --epochs 3 --output output/week5/three_epochs
~~~

Completed experiments are skipped on an identical rerun. An interrupted
experiment restarts from its first epoch. Completed result files are
written after predictions and checkpoint saving. Reusing an output directory
with changed settings raises an error to avoid mixing incompatible results.
For source-code changes affecting an experiment, use a fresh output folder.

You can also launch the package directly from the repository root:

~~~powershell
& $projectPython -m week5.run --quick
~~~

From inside the week5 folder, direct launching also works:

~~~powershell
& $projectPython run.py --quick
~~~

Completed runs retain their recorded environment versions. To measure results
after upgrading libraries, choose a fresh output directory so the new run
does not reuse historical models:

~~~powershell
& $projectPython Code_Week_2.py --output output/week5/latest
~~~

## Modules

| File | Purpose |
| --- | --- |
| `Code_Week_1.py` | Original Week 1 exploration and IMDb baseline |
| `Code_Week_2.py` | Week 5 entry point |
| `week5/run.py` | CLI, experiment configuration, and orchestration |
| `week5/common.py` | Cleaning, shared splits, metrics, atomic JSON outputs |
| `week5/baseline.py` | Binary/multiclass TF-IDF logistic regression |
| `week5/transformer.py` | Training-only vocabulary and scratch Transformer |
| `week5/training.py` | Neural training, validation selection, evaluation |
| `week5/transfer.py` | Pretrained BERT and four layer configurations |
| `week5/visualize.py` | Dataset plots, learning curves, Pareto frontiers |
| `week5/report.py` | Tables and measured comparisons for LaTeX |
| `main.tex` | Report methods and generated result inclusion |

## Outputs and report

Full results are stored under `output/week5/full`.
Smoke results use `output/week5/smoke` and are clearly marked as reduced-data results.
Each model folder contains metrics, classification reports, prediction CSVs,
and either a neural checkpoint or a baseline joblib file. Neural checkpoints
also store the model configuration, label metadata, and scratch vocabulary
or pretrained checkpoint identifier. Checkpoints use `state_dict`.
Transfer folders include tokenizer files and the BERT configuration.
The selected trainable tensor names and counts are recorded.
The run root contains `summary.csv`, `config.json`,
`environment.json`, and generated `results.tex`.

Compile the full report from the repository root:

~~~powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf main.tex
~~~

For a custom run, override the report directory:

~~~powershell
pdflatex -output-directory=output/pdf '\def\resultsdir{output/week5/three_epochs}\input{main.tex}'
~~~

The resulting report is `output/pdf/main.pdf`.
Missing result files are identified in the document. Figures are loaded
directly from the selected run folder. Head-only, first-block, last-block,
and full-network settings use the same pretrained initialization and seed.
Both validation and descriptive test Pareto plots are saved.
The test set does not select the checkpoint or best configuration.
The default study uses one seed and a fixed learning rate, so conclusions
apply to the tested configurations and budget.
