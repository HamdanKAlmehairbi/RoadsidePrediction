# RoadsidePrediction

2D object perception with the **nuImages nuMini** dataset and **YOLO** in Jupyter.

## Setup

1. **Download the dataset** (tar file) from:  
   https://drive.google.com/file/d/1y1yww20So4sYzAEjxbQxPGZqHhrcon46/view?usp=sharing

2. **Extract** the archive (e.g. unzip or use 7-Zip / tar).

3. **Place** the `nuimages-v1.0-mini` folder in the **same directory** as the Jupyter notebook (`nuimages_yolo_starter.ipynb`).

Your project folder should look like:

```
Project/
├── nuimages_yolo_starter.ipynb
├── nuimages-v1.0-mini/
│   ├── samples/
│   ├── sweeps/
│   └── v1.0-mini/
├── requirements.txt
└── README.md
```

4. Use **Python 3.9–3.12** (not 3.13). Create a conda env:  
   `conda create -n nuimages python=3.12` then `conda activate nuimages`.

5. Install dependencies:  
   `pip install -r requirements.txt`

6. Open `nuimages_yolo_starter.ipynb` and run the cells (run the pip cell once, then restart the kernel if needed).
