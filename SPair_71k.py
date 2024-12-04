import glob
import json
import os
import pickle
import random
import scipy.io

import numpy as np
#from PIL import Image

from config import cfg

import GM_GenData
from GM_GenData import *


SPAIR71K_CATEGORIES = cfg.SPair.CLASSES

# SPAIR71K_NODE_VISFEA_TYPE = GM_GenData.NODE_VISFEA_TYPE
SPAIR71K_NODE_VISFEA_TYPE = None
SPAIR71K_CNN_FEA_ROOT = '../LGM-data/vgg_features_cropped/SPair-71k'

cache_path = cfg.CACHE_PATH
pair_ann_path = cfg.SPair.ROOT_DIR + "/PairAnnotation"
layout_path = cfg.SPair.ROOT_DIR + "/Layout"
image_path = cfg.SPair.ROOT_DIR + "/JPEGImages"
dataset_size = cfg.SPair.size                       # large

sets_translation_dict = dict(train="trn", test="test")
difficulty_params_dict = dict(
    trn=cfg.TRAIN.difficulty_params, val=cfg.EVAL.difficulty_params, test=cfg.EVAL.difficulty_params
)


class SPair71k:
    def __init__(self, sets, obj_resize):
        """
        :param sets: 'train' or 'test'
        :param obj_resize: resized object size
        """
        self.sets = sets_translation_dict[sets]
        self.ann_files = open(os.path.join(layout_path, dataset_size, self.sets + ".txt"), "r").read().split("\n")
        self.ann_files = self.ann_files[: len(self.ann_files) - 1]
        cls_list=[]
        for ann_file in self.ann_files:
            if ann_file.split(":")[-1] not in cls_list:
                cls_list.append(ann_file.split(":")[-1])
        self.difficulty_params = difficulty_params_dict[self.sets]
        self.pair_ann_path = pair_ann_path
        self.image_path = image_path
        self.classes = list(map(lambda x: os.path.basename(x), glob.glob("%s/*" % image_path)))
        self.classes.sort()
        self.obj_resize = obj_resize
        self.combine_classes = cfg.combine_classes
        self.ann_files_filtered, self.ann_files_filtered_cls_dict, self.classes = self.filter_annotations(
            self.ann_files, self.difficulty_params
        )
        self.total_size = len(self.ann_files_filtered)
        self.size_by_cls = {cls: len(ann_list) for cls, ann_list in self.ann_files_filtered_cls_dict.items()}

    def filter_annotations(self, ann_files, difficulty_params):
        if len(difficulty_params) > 0:
            basepath = os.path.join(self.pair_ann_path, "pickled", self.sets)
            if not os.path.exists(basepath):
                os.makedirs(basepath)
            difficulty_paramas_str = self.diff_dict_to_str(difficulty_params)
            try:
                filepath = os.path.join(basepath, difficulty_paramas_str + ".pickle")
                ann_files_filtered = pickle.load(open(filepath, "rb"))
                print(
                    f"Found filtered annotations for difficulty parameters {difficulty_params} and {self.sets}-set at {filepath}"
                )
            except (OSError, IOError) as e:
                print(
                    f"No pickled annotations found for difficulty parameters {difficulty_params} and {self.sets}-set. Filtering..."
                )
                ann_files_filtered_dict = {}

                for ann_file in ann_files:
                    with open(os.path.join(self.pair_ann_path, self.sets, ann_file + ".json")) as f:
                        annotation = json.load(f)
                    diff = {key: annotation[key] for key in self.difficulty_params.keys()}
                    diff_str = self.diff_dict_to_str(diff)
                    if diff_str in ann_files_filtered_dict:
                        ann_files_filtered_dict[diff_str].append(ann_file)
                    else:
                        ann_files_filtered_dict[diff_str] = [ann_file]
                total_l = 0
                for diff_str, file_list in ann_files_filtered_dict.items():
                    total_l += len(file_list)
                    filepath = os.path.join(basepath, diff_str + ".pickle")
                    pickle.dump(file_list, open(filepath, "wb"))
                assert total_l == len(ann_files)
                print(f"Done filtering. Saved filtered annotations to {basepath}.")
                ann_files_filtered = ann_files_filtered_dict[difficulty_paramas_str]
        else:
            print(f"No difficulty parameters for {self.sets}-set. Using all available data.")
            ann_files_filtered = ann_files

        ann_files_filtered_cls_dict = {
            cls: list(filter(lambda x: cls in x, ann_files_filtered)) for cls in self.classes
        }
        class_len = {cls: len(ann_list) for cls, ann_list in ann_files_filtered_cls_dict.items()}
        print(f"Number of annotation pairs matching the difficulty params in {self.sets}-set: {class_len}")
        if self.combine_classes:
            cls_name = "combined"
            ann_files_filtered_cls_dict = {cls_name: ann_files_filtered}
            filtered_classes = [cls_name]
            print(f"Combining {self.sets}-set classes. Total of {len(ann_files_filtered)} image pairs used.")
        else:
            filtered_classes = []
            for cls, ann_f in ann_files_filtered_cls_dict.items():
                if len(ann_f) > 0:
                    filtered_classes.append(cls)
                else:
                    print(f"Excluding class {cls} from {self.sets}-set.")
        return ann_files_filtered, ann_files_filtered_cls_dict, filtered_classes

    def diff_dict_to_str(self, diff):
        diff_str = ""
        keys = ["mirror", "viewpoint_variation", "scale_variation", "truncation", "occlusion"]
        for key in keys:
            if key in diff.keys():
                diff_str += key
                diff_str += str(diff[key])
        return diff_str

    def get_k_samples(self, idx, k, mode, cls=None, shuffle=True):
        """
        Randomly get a sample of k objects from VOC-Berkeley keypoints dataset
        :param idx: Index of datapoint to sample, None for random sampling
        :param k: number of datapoints in sample
        :param mode: sampling strategy
        :param cls: None for random class, or specify for a certain set
        :param shuffle: random shuffle the keypoints
        :return: (k samples of data, k \choose 2 groundtruth permutation matrices)
        """
        if k != 2:
            raise NotImplementedError(
                f"No strategy implemented to sample {k} graphs from SPair dataset. So far only k=2 is possible."
            )

        if cls is None:
            cls = self.classes[random.randrange(0, len(self.classes))]
            ann_files = self.ann_files_filtered_cls_dict[cls]
        elif type(cls) == int:
            cls = self.classes[cls]
            ann_files = self.ann_files_filtered_cls_dict[cls]
        else:
            assert type(cls) == str
            ann_files = self.ann_files_filtered_cls_dict[cls]

        # get pre-processed images

        assert len(ann_files) > 0
        if idx is None:
            ann_file = random.choice(ann_files) + ".json"
        else:
            ann_file = ann_files[idx] + ".json"
        with open(os.path.join(self.pair_ann_path, self.sets, ann_file)) as f:
            annotation = json.load(f)

        category = annotation["category"]
        if cls is not None and not self.combine_classes:
            assert cls == category
        assert all(annotation[key] == value for key, value in self.difficulty_params.items())

        if mode == "intersection":
            assert len(annotation["src_kps"]) == len(annotation["trg_kps"])
            num_kps = len(annotation["src_kps"])
            perm_mat_init = np.eye(num_kps)
            anno_list = []
            perm_list = []

            for st in ("src", "trg"):
                if shuffle:
                    perm = np.random.permutation(np.arange(num_kps))
                else:
                    perm = np.arange(num_kps)
                kps = annotation[f"{st}_kps"]
                img_path = os.path.join(self.image_path, category, annotation[f"{st}_imname"])
                img, kps = self.rescale_im_and_kps(img_path, kps)
                kps_permuted = [kps[i] for i in perm]
                anno_dict = dict(image=img, keypoints=kps_permuted)
                anno_list.append(anno_dict)
                perm_list.append(perm)

            perm_mat = perm_mat_init[perm_list[0]][:, perm_list[1]]
        elif mode == "annotation":
            anno_list = None
            perm_mat = None
        else:
            raise NotImplementedError(f"Unknown sampling strategy {mode}")

        return annotation, anno_list, [perm_mat]

    def rescale_im_and_kps(self, img_path, kps):

        with Image.open(str(img_path)) as img:
            w, h = img.size
            img = img.resize(self.obj_resize, resample=Image.BICUBIC)

        keypoint_list = []
        for kp in kps:
            x = kp[0] * self.obj_resize[0] / w
            y = kp[1] * self.obj_resize[1] / h
            keypoint_list.append(dict(x=x, y=y))

        return img, keypoint_list

    def get_num_of_samples(self, category = None):
        if category is None:
            num = len(self.ann_files_filtered)
        else:
            num = len(self.ann_files_filtered_cls_dict[category])
        return num


def _list_images(root, category) :
    imgFiles = []
    path = os.path.join(root, category)
    for file in os.listdir(path) :
        filepath = os.path.join(path, file)
        if os.path.isfile(filepath) :
            if os.path.basename(filepath).endswith('.jpg') :
                imgFiles.append(file)

    return imgFiles

def _list_images_all_categories(root):
    imgFiles_all = []
    for category in SPAIR71K_CATEGORIES:
        imgFiles_all.append(_list_images(root, category))

    return imgFiles_all

def _load_cnn_feature(set_dict, filename):
    mat_file = SPAIR71K_CNN_FEA_ROOT + '/' + set_dict + '/' + filename  +  '.mat'

    if os.path.exists(mat_file):
        infos = scipy.io.loadmat(mat_file)
        pts0 = infos["pts_src"]
        pts1 = infos["pts_dst"]
        descs0 = infos["pts_features_src"].astype(np.float64)
        descs1 = infos["pts_features_dst"].astype(np.float64)
        descs0 = GM_GenData._mean_pooling(descs0, 32)
        descs1 = GM_GenData._mean_pooling(descs1, 32)
    else:
        raise(mat_file + ' is not exists !!!!!!!!!!!!')

    return pts0, descs0, pts1, descs1

def _load_keypoint_features(category, imgFile, pts, feaType):
    imgPath = "{}/{}/{}".format(image_path, category, imgFile)

    if feaType == VISFEA_TYPE_RAWPIXEL:
        patches = GM_GenData._compute_image_patches(imgPath, pts)
        descs = np.reshape(patches, (patches.shape[0], -1))
        descs = descs.astype(np.float64)
    else:
        raise("unknown feature type!")

    return pts, descs


def _precompute_all_features(feaType):
    feature_categories = {}
    for cls in range(len(SPAIR71K_CATEGORIES)):
        category = SPAIR71K_CATEGORIES[cls]
        feature_categories[category] = []

        imgFiles = _list_images(image_path, category)

        features = {}
        for file in imgFiles:
            pts, descs = _load_keypoint_features(category, file, feaType)

            feature = {"pts": pts, "descs": descs, "imname": file}
            features[file] = feature

        feature_categories[category] = features

    return feature_categories

def _preload_all_samples(dataset, sets_dict, feaType):
    sample_categories = {}
    for cls in range(len(SPAIR71K_CATEGORIES)):
        category = SPAIR71K_CATEGORIES[cls]
        num_samples = dataset.get_num_of_samples(category)

        samples = []

        for i in range(num_samples):
            annotation, anno_list, perm_mat = dataset.get_k_samples(i, 2, "annotation", cls)
            filename = annotation["filename"]
            src_img = annotation["src_imname"]
            trg_img = annotation["trg_imname"]
            src_pts = np.array(annotation["src_kps"])
            trg_pts = np.array(annotation["trg_kps"])

            pts0, descs0, pts1, descs1 = _load_cnn_feature(sets_dict, filename)

         #   src_pts, src_descs = _load_keypoint_features(category, src_img, src_pts, feaType)
         #   trg_pts, trg_descs = _load_keypoint_features(category, trg_img, trg_pts, feaType)
            sample = {"sets_dict": sets_dict,
                      "filename": filename,
                      "src_imname": src_img,
            #          "src_pts": src_pts,
            #          "src_descs": None,
                      "trg_imname": trg_img,
            #          "trg_pts": trg_pts,
            #          "trg_descs": None}
                      "pts0": pts0,
                      "pts1": pts1,
                      "descs0": descs0,
                      "descs1": descs1}
            samples.append(sample)

        sample_categories[category] = samples

    return sample_categories

dataset_train = SPair71k("train", (256, 256))
dataset_test =  SPair71k("test", (256, 256))
#features_categories = _precompute_all_features(SPAIR71K_NODE_VISFEA_TYPE)
samples_train = _preload_all_samples(dataset_train, "trn", SPAIR71K_NODE_VISFEA_TYPE)
samples_test = _preload_all_samples(dataset_test, "test", SPAIR71K_NODE_VISFEA_TYPE)

def _normalize_coordinates(points) :
    # normalize by center
    center = np.sum(points, axis = 0) / points.shape[0]
    norm_points = np.transpose(points)
    norm_points[0] = norm_points[0] - center[0]
    norm_points[1] = norm_points[1] - center[1]

    # normalized by max_distance
    distance = spatial.distance.cdist(points, points)
    maxDst = np.max(distance)
    norm_points = norm_points / maxDst

    if maxDst <= 0.0:
        print(points.shape)
        raise("invalid maxDst")

    # # normalize by deviation
    # deviation = np.nanstd(norm_points, axis=1)
    # norm_points[0] = norm_points[0] / deviation[0]
    # norm_points[1] = norm_points[1] / deviation[1]

    points = np.transpose(norm_points)

    return points

def _gen_random_graph(rand,
                      use_train_set,
                      category_id,
                      num_outlier_min_max,
                      feaType):
    # if category_id < 0:
    #     cls = None
    # else:
    #     cls = category_id
    #
    # if use_train_set:
    #     annotation, anno_list, perm_mat = dataset_train.get_k_samples(None, 2, "intersection", cls)
    # else:
    #     annotation, anno_list, perm_mat = dataset_test.get_k_samples(None, 2, "intersection", cls)
    # anno_pts0, anno_descs0, anno_pts1, anno_descs1 = _load_keypoint_features(annotation, feaType)
    #
    # pts0 = anno_pts0.copy()
    # pts1 = anno_pts1.copy()
    # descs0 = anno_descs0.copy()
    # descs1 = anno_descs1.copy()

    category = SPAIR71K_CATEGORIES[category_id]

    while True:
        if use_train_set:
            sample = random.choice(samples_train[category])
        else:
            sample = random.choice(samples_test[category])

    #    pts0, descs0, pts1, descs1 = _load_cnn_feature(sample["sets_dict"], sample["filename"])
        pts0 = sample["pts0"]
        pts1 = sample["pts1"]
        descs0 = sample["descs0"]
        descs1 = sample["descs1"]

        if pts0.shape[0] > 1 and pts1.shape[0] > 1:
            break

#    pts0, descs0 = _load_keypoint_features(category, sample["src_imname"], sample["src_pts"], feaType)
#    pts1, descs1 = _load_keypoint_features(category, sample["trg_imname"], sample["trg_pts"], feaType)

    # randomly re-order
    index0 = np.arange(0, pts0.shape[0])
    rand.shuffle(index0)
    pts0 = pts0[index0]
    descs0 = descs0[index0]

    index1 = np.arange(0, pts1.shape[0])
    rand.shuffle(index1)
    pts1 = pts1[index1]
    descs1 = descs1[index1]

    # normalize point coordinates
    # pts0 = GM_GenData._normalize_coordinates(pts0)
    # pts1 = GM_GenData._normalize_coordinates(pts1)
    pts0 = _normalize_coordinates(pts0)
    pts1 = _normalize_coordinates(pts1)

    # record ground-truth matches
    #    gX = np.eye(pts1.shape[0])
    gX = np.zeros((pts0.shape[0], pts1.shape[0]))
    for i in range(pts0.shape[0]):
        gX[i][i] = 1.0
    gX = np.transpose(np.transpose(gX[index0])[index1])

    # A0 = GM_GenData._build_delaunay_graph(pts0)
    # A1 = GM_GenData._build_delaunay_graph(pts1)
    A0 = GM_GenData._build_KNN_graph(pts0)
    A1 = GM_GenData._build_KNN_graph(pts1)

    filter = np.ones(shape=gX.shape, dtype=np.bool)

    graph = {"A0": A0,
             "A1": A1,
             "Filter": filter,
             "NodeFea0": descs0,
             "NodeFea1": descs1,
             "EdgeFea0": pts0,
             "EdgeFea1": pts1,
             "gX": gX}

    image = {"category": category,
             "image1": sample["src_imname"],
             "image2": sample["trg_imname"]}

    return graph, image


def gen_random_graphs_SPair71K(rand,
                              num_examples,
                              num_inner_min_max,
                              num_outlier_min_max,
                              feaType,
                              use_train_set = True,
                              category_id = -1):

    graphs = []
    images = []
    for _ in range(num_examples):
        if category_id < 0:
            cid = rand.randint(0, len(SPAIR71K_CATEGORIES))
        else:
            cid = category_id

        graph, image = _gen_random_graph(rand,
                                         use_train_set,
                                         cid,
                                         num_outlier_min_max=num_outlier_min_max,
                                         feaType = feaType)
        graphs.append(graph)
        images.append(image)

        #loss_conses = loss_conses + graph["loss_conses"]

#    loss_conses = loss_conses / num_examples
#    print("loss_conses = " + str(loss_conses))

    return graphs, images


if __name__ == "__main__":
    trn_dataset = SPair71k("train", (256, 256))

    annotation, anno_list, perm_mat = trn_dataset.get_k_samples(None, 2, "intersection")

    print(anno_list)
    print(perm_mat)
