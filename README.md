# Mixed Attention Network for Cross-domain Sequential Recommendation

This is our TensorFlow implementation for the paper:

Mixed Attention Network for Cross-domain Sequential Recommendation.


The code is tested under a Linux desktop with TensorFlow 1.12.3 and Python 3.6.8.



## Data Pre-processing



The script is `reco_utils/dataset/sequential_reviews.py` which can be excuted via:

```
python examples/00_quick_start/sequential.py --is_preprocessing True
```

  

## Model Training

To train our model on `Amazon` dataset (with default hyper-parameters): 

```
python examples/00_quick_start/sequential.py
```

## Misc

The implemention of self attention is modified based on *[TensorFlow framework of Microsoft](https://github.com/microsoft/recommenders)*.
