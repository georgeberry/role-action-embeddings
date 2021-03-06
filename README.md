# Code for paper on [role action embeddings](https://arxiv.org/pdf/1811.08019.pdf)

## Please cite as

Berry, George. "Role action embeddings: scalable representation of network positions." arXiv preprint arXiv:1811.08019 (2018).

## Notes

This was created using Python 3, it may not work with Python 2. All packages were installed using Anaconda's [miniconda distribution](https://conda.io/miniconda.html).

We include slightly modified versions of the [node2vec](https://github.com/aditya-grover/node2vec) and [GraphWave](https://github.com/snap-stanford/graphwave) repos for comparison.

We use model loading code from [Kipf and Welling](https://github.com/tkipf/gcn).

Graph classification datasets were downloaded [here](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets).

Node classification datasets can be found [here](https://linqs.soe.ucsc.edu/data), but we use the specific splits provided [here](https://github.com/tkipf/gcn).

Model code was adapted from [Hamilton et al's code](https://github.com/williamleif/graphsage-simple).

Make sure you have the dependencies installed. These are: `scikit-learn`, `numpy`, `pandas`, `networkx`, `matplotlib`, `seaborn`, `pytorch`, `gensim`, `jupyter`, `scipy`

## How to run `small_graphs.ipynb` code
- Open a Jupyter notebook with `jupyter notebook` at the command line and open `small_graphs.ipynb`

## How to run code in the `empirical_graphs` folder
- There are four notebooks, one for Cora, one for Citeseer, one for Pubmed, and one for all of the graph classification tasks
- Unzip `empirical_graphs/data.zip`
-  `empirical_graphs/models.py`, you should set the `base_data_path` variable to point to the `data` folder on your machine. Data is contained in the `empirical_graphs/data` folder
- For the graph classification, the `REDDIT-MULTI-12k` files are quite large, please download it from [here](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets).

## What are these `old-` files doing?
- They're leftovers from my original runs to generate the results in the paper. I then went back and cleaned up the code. You can ignore them, but I wanted to keep them around for posterity.
