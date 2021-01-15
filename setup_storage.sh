#!/bin/bash

# Run this interactively beforehand:
# pip install gsutil
# gsutil config

mkdir data_derived
mkdir data_derived/zipped
gsutil -m cp -r gs://vpl-bucket/data_derived/* ./data_derived/zipped
unzip crcns-mt2.zip -d data_derived
unzip crcns-pvc4.zip -d data_derived
gsutil -m cp -r gs://vpl-bucket/checkpoints/* ./checkpoints