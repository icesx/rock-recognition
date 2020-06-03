from dataset.image_file import *


class DatasetCreator:
    image_label: ImageLabels

    def __init__(self, image_x=64, image_y=64):
        self.image_x = image_x
        self.image_y = image_y
        self.ds = None
        self.image_label = None

    def load(self, root):
        return self.__create_dataset(root)

    def batch(self, batch):
        return self.ds.batch(batch), self.image_label

    def repeat(self):
        self.ds = self.ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=1024))
        return self

    def __load_and_preprocess_from_path_label(self, path, lable):
        images = image_byte_array(path, self.image_x, self.image_y)
        print("Preprocess Image:", path, lable)
        return images, lable

    def __create_dataset(self, root):
        image_label = image_labels(root)
        self.image_label = image_label
        ds = tf.data.Dataset.from_tensor_slices((image_label.paths, image_label.label_idx))
        self.ds = ds.map(self.__load_and_preprocess_from_path_label)
        # self.ds = ds.cache()
        return self


if __name__ == '__main__':
    ds = DatasetCreator().load('/WORK/datasset/rock_imgs_train2', 15)
    for element in ds:
        print(element)
