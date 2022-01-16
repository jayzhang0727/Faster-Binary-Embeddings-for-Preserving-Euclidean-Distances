# Faster Binary Embeddings for Preserving Euclidean Distances

Python code of our paper ["Faster Binary Embeddings for Preserving Euclidean Distances"](https://openreview.net/forum?id=YCXrx6rRCXO), 
accepted by ICLR 2021. In this paper, we propose a fast binary embedding algorithm to preserve Euclidean distances among well-spread vectors and 
it achieves optimal bit complexity. 

If you use our result, please cite our paper:

    @inproceedings{zhang2020faster,
     title={Faster Binary Embeddings for Preserving Euclidean Distances},
     author={Zhang, Jinjie and Saab, Rayan},
     booktitle={International Conference on Learning Representations},
     year={2020}
     }

### Description
quantizer.py contains all Sigma-Delta quantization schemes used in the paper. One can reproduce our numerical 
experiments by implementing Jupyter Notebook main.ipynb

### Requirements
- numpy
- scipy
- imageio
- matplotlib
- cv2
- skimage
- glob

### Datasets 
The following public datasets are used in our paper. The user need to get access to them before perform our code.

- [Yelp Dataset](https://www.yelp.com/dataset)
- [ImageNet](http://www.image-net.org/)
- [Flickr30k](https://github.com/BryanPlummer/flickr30k_entities)
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
