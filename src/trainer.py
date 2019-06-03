# Modified from training program in keras-retinanet with simplified settings.

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import argparse
import keras
import tensorflow as tf
# import from keras-retinanet
from keras_retinanet import models
from keras_retinanet import losses
from keras_retinanet.callbacks import RedirectModel
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.models.retinanet import retinanet_bbox
from keras_retinanet.utils.anchors import AnchorParameters, make_shapes_callback
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.utils.transform import random_transform_generator

# Support utility functions
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

class IdentifierTrainer():
    def __init__(self, backbone="resnet50", random_transform=False, batch_size=2, step=100,
                 epoch=40, anchor_settings=None):
        """ Initializer
            Args:
                backbone
                    string indicate which backbone network to use (only for ones defined in
                    keras-retinanet)
                random_transform
                    boolean indicate whether image is randomly transformed when training
                batch_size
                    number of images sent for each step during training
                step
                    number of steps(batches) to run for each epoch during training
                epoch
                    number of epochs for this training
                anchor_settings
                    a list of lists indicate anchor box parameters. please reference
                    keras-retinanet's document for how to set up anchor box parameters
        """
        check_keras_version()
        self._backbone_name = backbone
        self._backbone = models.backbone(backbone)
        self._img_preprocessor = self._backbone.preprocess_image
        self._batch_size = batch_size
        self._step = step
        self._epoch = epoch

        if random_transform:
            self._transform_generator = random_transform_generator(
                min_rotation=-0.1,
                max_rotation=0.1,
                min_translation=(-0.1, -0.1),
                max_translation=(0.1, 0.1),
                min_shear=-0.1,
                max_shear=0.1,
                min_scaling=(0.9, 0.9),
                max_scaling=(1.1, 1.1),
                flip_x_chance=0.5,
                flip_y_chance=0.5,
            )
        else:
            self._transform_generator = random_transform_generator(flip_x_chance=0.5)
 
        self._common_args = {
            'batch_size'       : batch_size,
            'preprocess_image' : self._img_preprocessor
        }

        if anchor_settings:
            self._anchor_params = AnchorParameters(*anchor_settings)
        else:
            self._anchor_params = AnchorParameters.default
            # # These settings are for DNA strands (narrow shapes)
            # self._anchor_params = AnchorParameters(
            #    [16, 32, 64, 128, 256],
            #    [8, 16, 32, 64, 128],
            #    [7.5, 10, 12.5, 15, 17.5, 20, 25, 30, 40, 50],
            #    [0.75, 1, 1.2, 1.4, 1.6]
            #)


    def __create_generator(self, annotation_path, class_path):
        train_generator = CSVGenerator(
            annotation_path,
            class_path,
            transform_generator=self._transform_generator,
            **(self._common_args)
        )
        return train_generator

    def __create_models(self, backbone_retinanet, num_classes):
        anchor_params = self._anchor_params
        num_anchors = anchor_params.num_anchors()
        model = backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=None)
        training_model = model
        prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)
        training_model.compile(
            loss={
                'regression'    : losses.smooth_l1(),
                'classification': losses.focal()
            },
            optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
        )

        return model, training_model, prediction_model

    def __create_callbacks(self, model, training_model, prediction_model, tensorboard_dir,
                           snapshot_dir, snapshot_tag, snapshot_freq):
        callbacks = []

        if tensorboard_dir:
            tensorboard_callback = keras.callbacks.TensorBoard(
                log_dir=tensorboard_dir, histogram_freq=0, batch_size=self._batch_size,
                write_graph=True, write_grads=False, write_images=False,
                embeddings_freq=0, embeddings_layer_names=None,
                embeddings_metadata=None
            )
            callbacks.append(tensorboard_callback)
        if snapshot_dir:
            makedirs(snapshot_dir)
            s_tag = snapshot_tag + "_" if snapshot_tag else ""
            checkpoint = keras.callbacks.ModelCheckpoint(
                os.path.join(
                    snapshot_dir,
                    '{tag}{backbone}_{{epoch:02d}}.h5'.format(tag=s_tag, backbone=self._backbone_name)
                ),
                verbose=1,
                period=snapshot_freq
            )
            checkpoint = RedirectModel(checkpoint, model)
            callbacks.append(checkpoint)
 
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.1, patience=2, verbose=1,
            mode='auto', min_delta=0.0001, cooldown=0, min_lr=0
        ))
        return callbacks

    def train(self, annotation_path, class_path, snapshot_src=None, tensorboard_dir=None,
              snapshot_dir=None, snapshot_tag=None, snapshot_freq=10):
        """ Train routine.
            Params:
                annotation_path: file path for target annotations
                class_path: file path for class-id mapping
                snapshot_src: file path of snapshot to continue training
                tensorboard_dir: directory for saving tensorboard
                snapshot_dir: directory for saving snapshots
                shapshot_tag: tag used as filename identifier for snapshots
                shapshot_freq: epoch interval for saving snapshots
        """
        if self._epoch % snapshot_freq != 0:
            print("Error: Snapshot saving interval should be set to factor of total epochns ({}).".format(self._epoch))
            return

        # Load network backbone
        generator = self.__create_generator(annotation_path, class_path)
 
        if snapshot_src:
            model = models.load_model(snapshot_src, backbone_name=self._backbone_name)
            training_model = model
            prediction_model = retinanet_bbox(model=model, anchor_params=self._anchor_params)
        else:
            model, training_model, prediction_model = self.__create_models(
                self._backbone.retinanet, generator.num_classes()
            )

        print(model.summary())

        if 'vgg' in self._backbone_name or 'densenet' in self._backbone_name:
            generator.compute_shapes = make_shapes_callback(model)

        callbacks = self.__create_callbacks(
            model,
            training_model,
            prediction_model,
            tensorboard_dir,
            snapshot_dir,
            snapshot_tag,
            snapshot_freq
        )
        training_model.fit_generator(
            generator=generator,
            steps_per_epoch=self._step,
            epochs=self._epoch,
            verbose=1,
            callbacks=callbacks
        )

def parse_args():
    parser = argparse.ArgumentParser(description='RetinaNet trainer.')
    parser.add_argument("annotations", type=str, help="File for annotation labels.")
    parser.add_argument("classes", type=str, help="File for id-class mapping.")
    parser.add_argument("label", type=str, help="CSV file for annotation labels.")
    parser.add_argument("board_dir", type=str, help="Folder for storing training evaluation.")
    parser.add_argument("snapshot_dir", type=str, help="Folder for storing training snaphsots.")
    parser.add_argument("--batch", required=False, type=int, default=1, dest="batch", \
                        help="Batch size. (default: 1)")
    parser.add_argument("--step", required=False, type=int, default=100, dest="step", \
                        help="Step size for each epoch. (default: 100)")
    parser.add_argument("--epoch", required=False, type=int, default=30, dest="epoch", \
                        help="Number of training epochs to run for the whole trainig proces. (default: 30)")
    parser.add_argument("--startpoint", required=False, type=str, default=None, \
                        dest="startpoint", help="Snapshot file for continuing training process.")
    parser.add_argument("--snapshot-tag", required=False, type=str, default=None, \
                        dest="snapshot_tag", help="Prefix for snapshot files.")
    parser.add_argument("--shapshot-freq", required=False, type=int, default=1, dest="snapshot_freq", \
                        help="Generate snapshot for every N epochs. If not set up, then snapshot will be \
                        generated for each epoch.")
    return parser.parse_args()


def run(args):
    # Usage example
    trainer = IdentifierTrainer(batch_size=args.batch_size, step=args.step, epoch=args.epoch)
    trainer.train(args.annotations, args.classes, snapshot_src=args.startpoint, tensorboard_dir=args.board_dir,
              snapshot_dir=args.snapshot_dir, snapshot_tag=args.snapshot_tag, snapshot_freq=args.snapshot_freq)

if __name__ == "__main__":
    args = parse_args()
    run(args)
