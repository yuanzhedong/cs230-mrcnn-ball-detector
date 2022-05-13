# MRCNN training

## Dataset

Data is being labeled using labelme and converted to coco

create classes.txt

```
__ignore__
ball
net
```

```
# start label
labelme  ~/Downloads/night/night_train --labels  ~/Downloads/night/classes.txt --nodata

# convert to coco
cd labelme/examples/instance_segmentation
./labelme2coco.py ~/Downloads/night/ ~/Downloads/night_labels --labels ~/Downloads/night/classes.txt 
```


### Training

```
bash run_docker.sh
python train.py
```




### Metric




### Reference

https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

https://linuxtut.com/en/a3be821734fd81c3ac59/
