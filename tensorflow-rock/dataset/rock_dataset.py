import tensorflow as tf
import pathlib
import random


class RockDataset:
    def __init__(self):
        pass

    def load(self,root):
        return self.__create_dataset(root)

    def __load_and_preprocess_from_path_label(self, path, label):
        image = tf.io.read_file(path)  # 读取图片
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [128, 128])
        image /= 255.0  # 归一化到[0,1]范围
        return image, label

    def __create_dataset(self, data_root_orig):
        data_root = pathlib.Path(data_root_orig)
        all_image_paths = list(data_root.glob('*/*'))
        print(all_image_paths)
        all_image_paths = [str(path) for path in all_image_paths]  # 所有图片路径的列表
        random.shuffle(all_image_paths)  # 打散

        label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
        print('label_names: ', label_names)  # ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

        label_to_index = dict((name, index) for index, name in enumerate(label_names))
        print('label_to_index: ',
              label_to_index)  # {'sunflowers': 3, 'daisy': 0, 'roses': 2, 'tulips': 4, 'dandelion': 1}

        all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
        print("First 10 labels indices: ", all_image_labels[:10])  # [2, 2, 2, 2, 3, 4, 1, 1, 3, 2]
        ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
        ds = ds.map(self.__load_and_preprocess_from_path_label)
        ds = ds.cache()
        ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=1024))
        return ds.batch(5)


if __name__ == '__main__':
    rd = RockDataset()
    ds = rd.load()
    for element in ds:
        print(element)
