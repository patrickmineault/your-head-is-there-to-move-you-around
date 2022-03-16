# Your head is there to move you around

This repo contains information necessary to run the code in "Your head is there to move you around". Preprint here: https://www.biorxiv.org/content/10.1101/2021.07.09.451701v2.abstract

# Prelims

## Python dependencies

Python 3.8. Create a new conda environment like so:

```
conda create --name yh python=3.8
conda activate yh
```

In this environment, `pip install -r requirements.txt`. You may use the Dockerfile if you prefer.

## Electrophysiology datasets

Data licenses do not allow us to openly redistribute derived datasets, hence the end-user will need to download datasets and preprocess them manually as shown below. 

### Preprocessed datasets

For review or reproducibility purposes, we can give you access to preprocessed datasets - just send me an email, and I will send you credentials to access the preprocessed datasets. Once that's done, you may download preprocessed datasets from Google Cloud as follows:

* `pip install gsutil`
* `mkdir zips`
* `gsutil rsync -r gs://vpl-bucket/data_derived zips`
* `unzip 'zips/*.zip' -d data_derived`

### Download raw datasets and preprocess

You can download the datasets `crcns-mt1`, `crcns-mt2`, `crcns-pvc1` and 
`crcns-pvc4` from [crcns.org](http://crcns.org/). You will need to register to 
do so. Drop them in a `raw_data` folder in folders that have the corresponding 
names (e.g. `crcns-mt1`, `crncs-mt2`, etc.). In `paths.py`, edit RAW_DATA and DERIVED_DATA to point to `raw_data` and `derived_data` folders. To preprocess the datasets:

* Run `scripts/generate_crncs_mt1_derived.m` in Matlab. You will need the 
  auxillary .m files that come with the dataset. Edit the 
  path in the m file. Then run `scripts/generate_crcns_mt1_derived.py`.
* Copy the contents of `crcns-mt2` verbatim into the `data_derived` folder
* Run `scripts/derive_dataset.py --dataset pvc1`
* Run `scripts/derive_dataset.py --dataset pvc4`

The preprocessing scripts for the `packlab-mst` dataset (Mineault et al. 2011) are organized similar to the crcns-mt1 data; the scripts are `generate_packlab_mst_derived.m` and `generate_packlab_mst_derived.py`.

## Airsim dataset

You can download [the exact dataset used in the paper on gdrive](https://drive.google.com/file/d/1P4vZhfs8OKOEqjxfwcUoWJ_N3nbZjXTD/view?usp=sharing), and drop it in `data_derived`. The code necessary to generate this dataset is in the `airsim` repo. It takes several days to run and requires some manual steps. 

## Checkpoints

[Download pretrained models from Google Drive](https://drive.google.com/file/d/16ABLAYyqc_fx7u6IZH0rXLnyYQbZy3HH/view?usp=sharing) and drop them in a `checkpoints` folder. We include CPC and DorsalNet checkpoints. Note that DorsalNet is often referred to in the code as `airsim_04`, since this was the fourth model we fit with the airsim dataset.

For the rest of the checkpoints, please [download the following from SlowFast](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md):

* `Kinetics/c2/SLOWFAST_8x8_R50`
* `Kinetics/c2/I3D_8x8_R50`

Edit `paths.py` CHECKPOINTS accordingly.

# Train DorsalNet

You can train DorsalNet on Airsim data using:

```
python train_sim.py 
  --exp_name trainit 
  --submodel dorsalnet_untied 
  --dataset airsim_batch2 
  --batch_size 64 
  --learning_rate 3e-3 
  --softmax 
  --decoder center 
  --num_epochs 100
```

This takes about 6 hours on a 1080 Titan X.

# Use pretrained DorsalNet

`notebooks/Use DorsalNet.ipynb` shows how to initialize and use DorsalNet with the checkpoint you downloaded.

# Estimate prefered stimuli

Use `notebooks/Show first layer filters.ipynb` to show first layer filters. Use `notebooks/Calculate standardized stimuli.ipynb` to calculate responses to gratings, plaids, etc. Use `scripts/dot_reverse_correlation.py` to calculate prefered motion vector fields. Run `notebooks/Show optimal stimuli.ipynb` to assemble quiver plots and calculate prefered image sequences using gradient descent. [See also here to visualize these optimal stimuli](https://flamboyant-babbage-94aa08.netlify.app/).

# Align deep neural nets to brain data

Align deep neural nets to brain data using `run_many.sh`. This should take a few weeks to run. You will need a wandb.ai account to save the results remotely.

# Compile brain data results

Use `notebooks/Compare results_physiology.ipynb` to assemble the brain data results into tables and plots.

# Heading decoding

Use `run_heading.sh` to run the heading decoding model. Use `notebooks/Compare results_heading.ipynb` to assemble results into tables and plots.

# Run experiments with scaling and boosting

You can run the experiments involving rescaling the data and aligning to the brain with boosting through `run_revision.sh`. This takes a few weeks to run.

