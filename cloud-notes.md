First, mirror data on the cloud using curl

```
output_dir = '../data/crcns/pvc4/zip'
files = ['crcns-pvc4-data.zip']

os.makedirs(output_dir)
!curl -d "username={crcns_username}&password={crcns_password}&fn=pvc-4/crcns-pvc4-data.zip&submit=Login" --output '{output_dir}/crcns-pvc4-data.zip' https://portal.nersc.gov/project/crcns/download/index.php
!unzip {output_dir}/{files[0]} -d ../crcns/pvc-4
```

Then, run a particular pipeline in the cloud

```
python train_net.py --exp_name _pvc4_pyramid_cell_00 --single_cell 00 --learning_rate 3e-3 --num_epochs 400 --nfeats 8 --warmup 1000


```