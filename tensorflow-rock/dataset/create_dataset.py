from dataset.image_file import *


class DatasetCreator:

    def __init__(self, image_x=64, image_y=64):
        self.image_x = image_x
        self.image_y = image_y
        self.ds = None

    def load(self, root):
        return self.__create_dataset(root)

    def batch(self, batch):
        self.ds = self.ds.batch(batch)
        return self

    def get(self):
        return self.ds

    def __load_and_preprocess_from_path_label(self, path):
        return image_byte_array(path, self.image_x, self.image_y)

    def __create_dataset(self, root):
        image_infos = image_labels(root)
        self.__ds_size = len(image_infos)
        path_ds = tf.data.Dataset.from_tensor_slices([ii.path_str for ii in image_infos])
        image_ds = path_ds.map(self.__load_and_preprocess_from_path_label)
        label_ds = tf.data.Dataset.from_tensor_slices(
            tf.cast([ii.label_info.label_idx for ii in image_infos], tf.int64))
        self.ds = tf.data.Dataset.zip((image_ds, label_ds))
        self.ds = self.ds.repeat()
        return self


if __name__ == '__main__':
    ds = DatasetCreator().load('/WORK/datasset/star_imgs_test').get()
    i = 0
    for element in ds:
        i += 1
        print("size ", i)
