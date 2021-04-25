from model.train_model import IDCardModel
import tensorflow as tf
if __name__ == '__main__':
    IDCardModel(image_root="/WORK/datasset/id_card/train", val_image_root="/WORK/datasset/id_card/val", image_y=256,
                image_x=32) \
        .load(batch=20, func_label=lambda ii: ii.label_info.label_name, label_type=tf.string) \
        .plot() \
        .train(steps_per_epoch=500,
               epochs=80) \
        .save("../save/model/star")
