# FindJellybeans
This project aims to demonstrate a brief example of how to extact target files as what's done in my previous collaborative course [project](https://github.com/tnmcneil/XGenomesProject).

- **Note**: It's still ongoing due to failure in previous early attempts, and I'll keep it updated, until reach some favorable result.

In the original work, we need to locate and annotate DNA strands on real images for further work. However the huge amount of target in the dataset makes it impossible to be manually done in just a few weeks. Fortunately we were able to apprximate and generate synthesized images, and train an object identification model on these images, so that we can thoses strands automatically. As the result, we saved much time on data preprocessing and brings an acceptable solution for the annotation problem.

Since we were colloborating on private genomic images (still you can see some result from report in the original repository), so I'm just using another case here instead, where both source for synthesized data and test data can be easily obtained through the net.

In this example, I'll show how to identify jelly beans and guess bean flavour in an image with [keras-retinanet](https://github.com/fizyr/keras-retinanet) package. The whole process starts with making some synthesized data to reduce effort in doing annotations manually, then use the synthesized images as training dataset to train our model with default settings. Finally we'll use real images to evaluate the final model.

## Prerequisites
All the codes are developed under `Python 3.6`'s environment (Note: `Tensorflow` does not support Python3.7 at this timepoint).

For data synthesizing part, you may need to install `opencv-python` with `pip` (`conda install -c conda-forge opencv` with `conda`) before starting off. To run the jupyter notebook, the most easiest way is to install [Anaconda](https://www.anaconda.com/) to get `Jupyter` built on your machine.

As for image identification part, follow the instructions [here](https://github.com/fizyr/keras-retinanet) here to install `keras-retinanet` package. You may need to install `numpy` and `tensorflow` through `pip` beforehand. (The repository readme also provide a list of troubleshooting instructions.)

## Instructions
You can read the workflow in `SynthesizeImage.ipynb` (you can read the code and output online without having `Jupyter`.) for how to make synthesized images, then you can collect some jelly bean images of your favorite and generate your own dataset.

At this point you'll have a set of synthesized images, annotation data, and mapping between class name and numeral ID. Then you can run train.py and start to train your model.

For real case prediction, please use the `convert_model.py` script provided in `keras-retinanet`'s repository first to obtain prediction model, then use the converted model for `predict.py`.

## Special Thanks
Thanks for [Yang'](https://github.com/tyku12cn)s contribution in the original project, for the idea to generate sythesized data.
