from keras_preprocessing.image import ImageDataGenerator

from dataset.image_file import image_byte_array, image_labels, ALL_LABELS
import tensorflow as tf
from utils.plot import PlotContext
from utils.tf_gpu import gpu_init


class DatasetCreator(object):
    def __init__(self, root, image_y=64, image_x=64):
        gpu_init(6000)
        print("DatasetCreator: init GPU.....")
        self._image_x = image_x
        self._image_y = image_y
        self.__ds = self._create_dataset(root)
        self.__plot_dataset()

    def __plot_dataset(self):
        pc = PlotContext(ncols=6, nrows=6)
        labels = dict((v.label_idx, v.label_name) for k, v in ALL_LABELS.items())
        print("ALL Labels:", labels)
        for image, label in self.__ds.take(36):
            pc.append(image, labels.get(label.numpy()))
        pc.show()

    def batch(self, batch):
        self.__ds = self.__ds.batch(batch)
        return self

    def get(self):
        return self.__ds

    def _create_dataset(self, root):
        return None


class CustomDatasetCreator(DatasetCreator):
    def __init__(self, root, image_y=64, image_x=64):
        DatasetCreator.__init__(self, root, image_y, image_x)

    def _create_dataset(self, root):
        image_infos = image_labels(root)
        path_ds = tf.data.Dataset.from_tensor_slices([ii.path_str for ii in image_infos])
        image_ds = path_ds.map(self.__load_and_preprocess_from_path_label)
        label_ds = tf.data.Dataset.from_tensor_slices(
            tf.cast([ii.label_info.label_idx for ii in image_infos], tf.int64))
        _ds = tf.data.Dataset.zip((image_ds, label_ds))
        return _ds.repeat().shuffle(buffer_size=1024)

    def __load_and_preprocess_from_path_label(self, path):
        return image_byte_array(path, self._image_x, self._image_y)


class IDGDatasetCreator(DatasetCreator):
    def __init__(self, root, image_y=64, image_x=64):
        DatasetCreator.__init__(self, root, image_y, image_x)

    def _create_dataset(self, root):
        return ImageDataGenerator(rescale=1. / 255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True) \
            .flow_from_directory(root,
                                 batch_size=32,
                                 target_size=(self._image_y, self._image_x))


if __name__ == '__main__':
    ds = CustomDatasetCreator('/WORK/datasset/star_imgs_test').get()
    i = 0
    for element in ds:
        i += 1
        print("size ", i)
