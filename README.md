# Activation functions for CNN (ImageNet)
## Activation functions for convolutional neural networks: proposals and experimental study


## Algorithms included

This repo contains the code to run experiments for ImageNet with four activation functions that have been used recently for convolutional network models. These are:

* ReLU
* ELU
* ELUs<sub>+2</sub>
* ELUs<sub>+2</sub>L

## Installation

### Dependencies

This repo basically requires:

 * Python         (>= 3.6.8)
 * munch          (== 2.5.0)
 * numpy          (== 1.19.2)
 * sacred         (== 0.8.2)
 * scikit-learn   (== 0.24.1)
 * tensorflow-gpu (== 2.3.0)
 * scikit-learn   (== 0.24.1)

### Compilation

To install the requirements, use:

**Install for GPU**
  `pip install -r requirements.txt`


## Development

Contributions are welcome. Pull requests are encouraged to be formatted according to [PEP8](https://www.python.org/dev/peps/pep-0008/), e.g., using [yapf](https://github.com/google/yapf).

## Usage

You can run the experiments with ImageNet dataset using:

  ```sh
  python train_imagenet224.py
  ```

## Citation

The paper titled "Activation functions for convolutional neural networks: proposals and experimental study" has been submitted to IEEE Transactions on Neural Networks and Learning Systems (IEEE TNNLS).

## Contributors

#### Activation functions for convolutional neural networks: proposals and experimental study

* Víctor Manuel Vargas ([@victormvy](https://github.com/victormvy))
* Pedro Antonio Gutiérrez ([@pagutierrez](https://github.com/pagutierrez))
* Javier Barbero Gómez ([@javierbg](https://github.com/javierbg))
* César Hervás-Martínez (chervas@uco.es)
