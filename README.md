# FindJellybeans
This project aims to demonstrate a brief example of how to extact target files as what's done in my previous collaborative course [project](https://github.com/tnmcneil/XGenomesProject).

- **Note**: It's still an ongoing project, and I'll keep it updated until reach some satisfactory result.

In the original work, we need to locate and annotate DNA strands on real images for further work (identify whether the extracted block contains specific DAN series). However the huge amount of target in the dataset makes it impossible to be manually done in just a few weeks. Fortunately we made it to apprximate and generate synthesized images, then trained an object identification model on these images and applied the identifier on real images for annotation. As the result, we saved much time on data preprocessing and brought an acceptable solution for the annotation problem.

Since we were colloborating on private genomic images (still you can see some result from report in the original repository), I'm just using another case here instead, where both source for synthesized data and test data can be easily obtained through the net. In this example, I'll show how to identify jelly beans in an image with [keras-retinanet](https://github.com/fizyr/keras-retinanet) package. The whole process starts with making some synthesized data to reduce effort in doing annotations manually, then use the synthesized images as training dataset to train our model with default settings. Finally we'll use real images to evaluate the trained model.

## Prerequisites
All the codes are developed under `Python 3.6`'s environment (Note: `Tensorflow` does not support Python 3.7 at this point).

For data synthesizing part, you may need to install `opencv-python` with `pip` (`conda install -c conda-forge opencv` with `conda`) before starting off. To run the jupyter notebook, the most easiest way is to install [Anaconda](https://www.anaconda.com/) to get `Jupyter` built on your machine.

As for image identification part, follow the instructions in official repository to install `keras-retinanet` package. You may need to install `numpy` and `tensorflow` through `pip` beforehand. The repository readme also provide a list of troubleshooting instructions, which can be helpful if you have any problem with using their package and scripts.

## Instructions
You can read the workflow in `SynthesizeImage.ipynb` to know how synthesized images can be generated, and then you can collect some jelly bean images of your favorite flavours and generate your own dataset.

Once you have a set of synthesized images, annotation data, and mapping between class name and numeral ID, you can run `/src/trainer.py` to start training your model.

For real case prediction, please use the `convert_model.py` script provided in `keras-retinanet`'s repository first to obtain the prdiction model, then use the it with `/src/predict.py`.

## Expected Result
![](/img/oneclass_predict_75_test_1.jpeg)
By following these instructions, you'll finally have a one-class identification model that's capable of pointing out where the jelly beans are. Though not perfect (you may consider tuning threshold parameter in `/src/predict.py` to get a better filtered result), and can fail in some real cases, we've already have a good start!

As for multiple-class identifier -- identify both location of jelly beans along with their flavor, unfortunately it turns out with the same result as when I was working on DNA strands -- several tries on different training settings are all getting poor result, and even can't point out location of the beans well. Apparaently there's still much improvement can be done for synthesizing, or perhaps sending the extracted blocks to another classifier for specific classes is usually a better idea for this task.

## Special Thanks
Thanks for [Yang'](https://github.com/yku12cn)s contribution in the original project, for the idea to generate sythesized data for automated annotation process.
