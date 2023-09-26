# TODO: replace with your folders and stored datasets

FOLDER=/cluster/scratch/denysr/dataset/
WEIGHTS=/cluster/scratch/denysr/tmp/
USERNAME=rozumden
scp $USERNAME@ptak.felk.cvut.cz:/datagrid/personal/rozumden/360photo/360photo.zip $FOLDER/360photo/
unzip $FOLDER/360photo/360photo.zip -d $FOLDER/360photo/

scp $USERNAME@ptak.felk.cvut.cz:/mnt/datasets/votrgbd/votrgbd.zip $FOLDER/votrgbd/
unzip $FOLDER/votrgbd/votrgbd.zip -d  $FOLDER/votrgbd/

wget http://ptak.felk.cvut.cz/public_datasets/coin-tracking/ctr.tar.gz -P $FOLDER/coin/
tar xvf $FOLDER/coin/ctr.tar.gz -C $FOLDER/coin/
scp -r $USERNAME@ptak.felk.cvut.cz:/mnt/datasets/coin-tracking/results/CTRBase $FOLDER/coin/results/
scp -r $USERNAME@radon.felk.cvut.cz:/ssd/export/D3S-masks/CTR/* $FOLDER/coin/results/D3S/

wget http://data.vicos.si/alanl/d3s/SegmNet.pth.tar -O $WEIGHTS/SegmNet.pth.tar
tar xvf  $WEIGHTS/SegmNet.pth.tar -C  $WEIGHTS
wget https://www.dropbox.com/s/hnv51iwu4hn82rj/s2dnet_weights.pth?dl=0 -O $FOLDER/s2dnet_weights.pth

# Download RAFT optical flow parameters
wget https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip -O $FOLDER/models.zip
unzip $FOLDER/models.zip -d  $FOLDER/raft_models

# Download OSTrack model parameters
scp $USERNAME@ptak.felk.cvut.cz:/mnt/datasets/votrgbd/ostrack_models.zip $FOLDER/ostrack/
unzip $FOLDER/ostrack/ostrack_models.zip -d  $FOLDER/ostrack/
mv $FOLDER/ostrack/models/* $FOLDER/ostrack/

# Download AlphaRefine (requires password)
scp $USERNAME@ptak.felk.cvut.cz:/mnt/datasets/votrgbd/SEcmnet_ep0040-c.pth.tar $FOLDER/ostrack/
