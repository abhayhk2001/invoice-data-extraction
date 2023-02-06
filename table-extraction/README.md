## Code Installation
Create a conda environment from the yml file and activate it as follows

```
cd table-extraction
conda env create -f env.yml
conda activate table-extraction
```

## Using Model for inference
```
$ python table-extractor.py <image_path> <table_detection_treshold> <table_detection_treshold> <padding_top> <padding_left> <padding_bottom> <padding_right>
```

# Fine-Tuning
Use the table-training repository to train the model and then replace the .bin file into the table-extraction\table-transformer-structure-recognition directory, and rename the file to pytorch_model.bin