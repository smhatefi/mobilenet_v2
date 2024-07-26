# MobileNet V2 Implementation from scratch
This repository contains the implementation of MobileNetV2 to classify images of cats and dogs using TensorFlow.

## Repository Structure
- `models/`: Contains the MobileNetV2 model definition.
- `utils/`: Contains utility functions for preprocessing and image handling.
- `train.py`: Script to train the model.
- `evaluate.py`: Script to evaluate the model.
- `predict.py`: Script to make predictions using the trained model.
- `requirements.txt`: List of dependencies.
- `mobilenet_v2_cats_vs_dogs.h5`: Pre-trained weights provided for your convenience.

## Usage
### Training
To train the model, run:
```
python train.py
```

### Evaluation
To evaluate the model, run:
```
python evaluate.py
```

### Prediction
Download your desired image and save it in the main directory of project with the name `example.jpg`
```
# For Example
wget -q -O example.jpg https://cdn-prod.medicalnewstoday.com/content/images/articles/322/322868/golden-retriever-puppy.jpg
```
Run the `predict.py` script to make predictions using the trained model:
```
python predict.py
```

### Dependencies
Install the required dependencies using:
```
pip install -r requirements.txt
```

## Dataset
The dataset used for training is the Cats vs. Dogs dataset from TensorFlow Datasets.
![Cats vs. Dogs dataset](https://github.com/smhatefi/mobilenet_v1/blob/main/cats_vs_dogs.png)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.


## Acknowledgments

The MobileNetV2 model architecture is inspired by the original [MobileNetV2 paper](https://arxiv.org/abs/1801.04381).
