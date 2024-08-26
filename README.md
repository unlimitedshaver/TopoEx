
1.Install from conda_env.txt

```
conda create --name tpex --file conda_env.txt
conda activate tpex
conda install -c conda-forge rdkit
pip install xxx [ogb,gudhi,pandas,...]
```

2.Install from scratch

```
pip install torch==1.13.1 --extra-index-url https://download.pytorch.org/whl/${CUDA}
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.1+${CUDA}.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.13.1+${CUDA}.html
```

where `CUDA` should be replaced by either `cpu`,`cu113`,`cu116`depending on your PyTorch installation `torch.version.cuda`

**Note: You should check the above code website to see if there is a corresponding cuda version**

The rest of the packages install themselves as described in the requirements.txt file
