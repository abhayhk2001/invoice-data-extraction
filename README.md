# invoice-data-extraction
Using ML and DL to extract information from documents mostly invoices on Windows platform.

# Prerequisites
Python 3.4+ installed globally.

1. [Git LFS](https://git-lfs.com/) - for installation of model weights
2. [Anaconda](https://docs.anaconda.com/anaconda/install/) - for setting up environment

Add conda to path or open the project using the Anaconda prompt.
But Anaconda is a heavy utility, you can choose miniconda.

[Install miniconda](https://docs.conda.io/en/main/), then add conda to path or open the project using the miniconda prompt

## Installing Tessaract OCR:
Install the latest version of [tessaract OCR](https://github.com/UB-Mannheim/tesseract/wiki) into the C directory and add the path (C:\Program Files \Tesseract-OCR) to both System and User environment variables in Windows. Download the additional eng_layer.traineddata file and add it to C:\Program Files\Tesseract-OCR\tessdata

## Install poppler
<!-- TODO:  -->

## Running the Code.
1. Clone the repository or downlaod the zip file from GitHub
```
 git clone https://github.com/abhayhk2001/document-data-extraction
```

2. Open a Terminal window in the same folder as the downloaded code.
Create a conda environment from the yml file and activate it as follows

```
conda env create -f env.yml
conda activate data-extraction
```

3. Add the invoice to examples subfolder.

4. To run the application.  
```
 python main.py --file [filename relative path]
```
5. Example

```
 python main.py --file examples\airtel_june_2012.pdf
```
6. Results are stored in results.txt and table.csv within runs/detect/exp* directories.