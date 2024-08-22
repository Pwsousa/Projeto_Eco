import os
import shutil
import random


def split_dataset(image_dir, train_dir, test_dir, split_ratio=0.8):
    # Cria diretórios para treinamento e teste
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Lista as classes no diretório original
    classes = os.listdir(image_dir)

    for class_name in classes:
        class_dir = os.path.join(image_dir, class_name)
        all_images = os.listdir(class_dir)
        random.shuffle(all_images)  # Embaralha as imagens

        # Divide as imagens entre treino e teste
        split_index = int(len(all_images) * split_ratio)
        train_images = all_images[:split_index]
        test_images = all_images[split_index:]

        # Diretórios para a classe
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(train_class_dir):
            os.makedirs(train_class_dir)
        if not os.path.exists(test_class_dir):
            os.makedirs(test_class_dir)

        # Move as imagens para os diretórios apropriados
        for img_name in train_images:
            shutil.copy(os.path.join(class_dir, img_name), os.path.join(train_class_dir, img_name))
        for img_name in test_images:
            shutil.copy(os.path.join(class_dir, img_name), os.path.join(test_class_dir, img_name))


# Caminhos para os diretórios
image_dir = 'dataset/'
train_dir = 'dataset/train/'
test_dir = 'dataset/test/'

split_dataset(image_dir, train_dir, test_dir)
