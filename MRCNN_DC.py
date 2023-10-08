"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------
"""

import os
import sys
import numpy as np
import cv2
# from natsort import natsorted

# Uncomment and set GPU id
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from config import Config
import model as modellib, utils

RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

global args

############################################################
#  Configurations
############################################################
class FundusConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "fundus"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + Disk + Cup

    #BATCH_SIZE = 8

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    VALIDATION_STEPS = 50

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.85

    USE_MINI_MASK = True

    #MINI_MASK_SHAPE = (224, 224)  # (height, width) of the mini-mask

    LEARNING_RATE = 0.01

    LEARNING_MOMENTUM = 0.9

    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 8

    TRAIN_ROIS_PER_IMAGE = 8

    BACKBONE = "resnet101"

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 1024

    MAX_GT_INSTANCES = 1

    TRAIN_BN = False

    IMAGE_CHANNEL_COUNT = 3

    # Image mean (RGB)
    MEAN_PIXEL = np.array([83.3, 49.67, 28.2])


############################################################
#  Dataset
############################################################
class FundusDataset(utils.Dataset):

    def load_fundus(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset vessel, and the class vessel
        self.add_class("fundus", 1, "disk")
        self.add_class("fundus", 2, "cup")

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val"]

        # Get directories containing input, disc, cup
        dataset_dir1 = dataset_dir
        dataset_dir = os.path.join(dataset_dir1, subset, "images")
        d_maskset_dir = os.path.join(dataset_dir1, subset, "OD")
        c_maskset_dir = os.path.join(dataset_dir1, subset, "OC")

        print("DataPath: ", dataset_dir)
        print("DiskMaskPath: ", d_maskset_dir)
        print("CupMaskPath: ", c_maskset_dir)

        # # Sort filenames in each directory
        image_ids = next(os.walk(dataset_dir), (None, None, []))[2]
        d_mask_ids = next(os.walk(d_maskset_dir), (None, None, []))[2]
        c_mask_ids = next(os.walk(c_maskset_dir), (None, None, []))[2]

        print(image_ids)
        print(d_mask_ids)
        print(c_mask_ids)

        print("\nLength: ", len(image_ids))
        i = 0

        # Add images
        for image_id in image_ids:

            if image_id != '.DS_Store':
                image = cv2.imread(os.path.join(dataset_dir, image_id))
                height, width = image.shape[:2]

                D_DIR = os.path.join(d_maskset_dir, d_mask_ids[i])
                C_DIR = os.path.join(c_maskset_dir, c_mask_ids[i])

                self.add_image(
                    "fundus",
                    d_mask_path=D_DIR,
                    c_mask_path=C_DIR,
                    width=width, height=height,
                    image_id=image_id,
                    path=os.path.join(dataset_dir, image_id))

                i = i + 1

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "fundus":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]

        # Read disc and cup mask for the given input image
        imD = cv2.imread(info["d_mask_path"], 0)
        imC = cv2.imread(info["c_mask_path"], 0)

        ret, thrD = cv2.threshold(imD, 0, 255, cv2.THRESH_BINARY)
        ret, thrC = cv2.threshold(imC, 0, 255, cv2.THRESH_BINARY)

        # Extract disc and cup boundaries
        contoursD, _ = cv2.findContours(thrD, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contoursC, _ = cv2.findContours(thrC, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        instance_masks = []
        class_ids = []

        mask = np.zeros([info["height"], info["width"], 1], dtype=np.uint8)
        blank = np.zeros(thrD.shape)
        mask = cv2.drawContours(blank, contoursD, 0, (1, 1, 1), -1)

        class_ids.append(1)
        instance_masks.append(mask)
        #blank = cv2.resize(mask, (1072, 712))
        #cv2.imshow("0", blank * 255)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        mask = np.zeros([info["height"], info["width"], 1], dtype=np.uint8)
        blank = np.zeros(thrC.shape)
        mask = cv2.drawContours(blank, contoursC, 0, (1, 1, 1), -1)

        class_ids.append(2)
        instance_masks.append(mask)
        #blank = cv2.resize(mask, (1072, 712))
        #cv2.imshow("0", blank * 255)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return np.stack(instance_masks, axis=2).astype(np.bool), np.array(class_ids, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "fundus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    global args
    """Train the model."""
    # Training dataset.
    dataset_train = FundusDataset()
    dataset_train.load_fundus(args["dataset"], "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FundusDataset()
    dataset_val.load_fundus(args["dataset"], "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("...Training network heads...")
    model.train(dataset_train, dataset_val,
              learning_rate=config.LEARNING_RATE,
              epochs=450,
              layers='all')

############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Read dataset
    dataset = FundusDataset()
    dataset.load_fundus(dataset_dir, subset)
    dataset.prepare()

    # Load over images
    for image_id in dataset.image_ids:

        # Load image and run detection
        image = dataset.load_image(image_id)

        # Detect objects
        r = model.detect([image], verbose=0)[0]

        d = np.zeros((image.shape[0], image.shape[1], 1))
        c = np.zeros((image.shape[0], image.shape[1], 1))

        N = r['rois'].shape[0]

        for i in range(N):

            # Bounding box
            if not np.any(r['rois'][i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue

            mask = r['masks'][:, :, i]

            mask = mask.reshape((mask.shape[0], mask.shape[1], 1))

            if r['class_ids'][i] == 1:
                d += mask

            elif r['class_ids'][i] == 2:
                c += mask

        fname = dataset.image_info[image_id]["id"][:-4] + ".png"

        print(fname)

        d = np.asarray(d, dtype = np.uint8)
        c = np.asarray(c, dtype = np.uint8)

        _, contoursD, _ = cv2.findContours(d, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        _, contoursC, _ = cv2.findContours(c, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        image = cv2.drawContours(image, contoursD, 0, (0, 255, 255), 2)
        image = cv2.drawContours(image, contoursC, 0, (255, 255, 0), 2)

        cv2.imshow("I", image)
        cv2.imshow("D", d)
        cv2.imshow("C", c)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("DONE")

############################################################
#  Prediction
############################################################
def predict(model, dataset_dir, OD_path, OC_path, save=True, show=False):

    image_ids = next(os.walk(dataset_dir))[2]

    print(image_ids)

    print("\nLength: ", len(image_ids))
    
    for img in image_ids:

        image = np.asarray(cv2.imread(os.path.join(dataset_dir, img)), dtype='float')

        image[..., 2] -= np.mean(image[..., 2])
        image[..., 1] -= np.mean(image[..., 1])
        image[..., 0] -= np.mean(image[..., 0])

        tmp = image[..., 2].copy()
        image[..., 2] = image[..., 0].copy()
        image[..., 0] = tmp

        # Detect objects
        r = model.detect([image], verbose=0)[0]

        d = np.zeros((image.shape[0], image.shape[1], 1))
        c = np.zeros((image.shape[0], image.shape[1], 1))

        N = r['rois'].shape[0]

        for i in range(N):

            # Bounding box
            if not np.any(r['rois'][i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue

            mask = r['masks'][:, :, i]

            mask = mask.reshape((mask.shape[0], mask.shape[1], 1))

            if r['class_ids'][i] == 1:
                d += mask

            elif r['class_ids'][i] == 2:
                c += mask

        print(img)

        #cv2.cvtColor(image, image, cv2.COLOR_BGR2RGB)
        #cv2.imwrite("./Results-Cross/HRF/images/" + img[:-4] + ".png", image)

        if save:
            cv2.imwrite(OD_path + "/" + img[:-4] + ".png", d * 255)
            cv2.imwrite(OC_path + "/" + img[:-4] + ".png", c * 255)

        if show:

            d = np.asarray(d, dtype = np.uint8)
            c = np.asarray(c, dtype = np.uint8)

            image = cv2.imread(os.path.join(dataset_dir, img))

            contoursD, _ = cv2.findContours(d, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contoursC, _ = cv2.findContours(c, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

            image = cv2.drawContours(image, contoursD, 0, (0, 255, 255), 2)
            image = cv2.drawContours(image, contoursC, 0, (255, 255, 0), 2)

            image = cv2.resize(image, (1200, 900))
            d = cv2.resize(d, (1200, 900))
            c = cv2.resize(c, (1200, 900))

            cv2.imshow("I", image)
            cv2.imshow("D", d)
            cv2.imshow("C", c)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print("DONE")
    print("Results saved in:\n", OD_path, "\n", OC_path)

############################################################
#  Training
############################################################

if __name__ == '__main__':

    # global args

    # Use this command for prediction
	# dataset: path to input RGB images
	# weights: trained weights file
	# OD_out: output directory of OD results
	# OC_out: output directory of OC results
    args = {"command":"predict", "dataset":"./Drishti/val/images", "weights":"./new_best_all.h5", "logs":DEFAULT_LOGS_DIR, "OD_out":"./OD", "OC_out":"./OC"}

    # Use this command for training
    #args = {"command":"train", "dataset":"./ALL", "weights":"./new_best_all.h5", "logs":DEFAULT_LOGS_DIR}

    # Validate arguments
    if args["command"] == "train":
        assert args["dataset"], "Argument --dataset is required for training"

    print("Weights: ", args["weights"])
    print("Dataset: ", args["dataset"])
    print("Logs: ", args["logs"])

    # Configurations
    if args["command"] == "train":
        config = FundusConfig()
    else:
        class InferenceConfig(FundusConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()

    # Display the config
    config.display()

    # Create model
    if args["command"] == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args["logs"])
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args["logs"])

    # Select weights file to load
    if args["weights"].lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args["weights"].lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args["weights"].lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args["weights"]

    # Load weights
    print("Loading weights ", weights_path)
    if args["weights"].lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args["command"] == "train":
        train(model)
    elif args["command"] == "detect":
        detect(model, args["dataset"], "train")
    elif args["command"] == "predict":
        if not os.path.exists(args["OD_out"]):
            os.makedirs(args["OD_out"])
            
        if not os.path.exists(args["OC_out"]):
            os.makedirs(args["OC_out"])
			
        predict(model, args["dataset"], args["OD_out"], args["OC_out"])
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
