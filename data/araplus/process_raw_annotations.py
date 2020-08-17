from pathlib import Path
from random import sample

from PIL import Image

flatten = lambda l: [item for sublist in l for item in sublist]
num_distinct_classes = 1
def box_transformer(box, width, height):
    box_class = 0 * int(box.pop(1))
    x_c = int(box[1] * width)
    y_c = int(box[2] * height)
    w = int(box[3] * width)
    h = int(box[4] * height)
    box[1] = max(x_c - w // 2, 0)
    box[2] = max(y_c - h // 2, 0)
    box[3] = min(box[1] + w + 1, width)
    box[4] = min(box[2] + h + 1, height)
    box.append('p{}'.format(box_class))

    if box[1] >= box[3] or box[2] >= box[4] or box[1]  < 0 or box[2] < 0 or box[3] > width or box[4] > height:
        print('invalid box', box, width, height)
    return box

def transform_group_anns(img_path, width, height):
    ann_path = img_path.with_suffix('.txt')
    f = open(ann_path, 'r')
    boxes = f.readlines()
    new_boxes = []
    for box in boxes:
        box = list((float(_) for _ in box.split()))
        box.insert(0, str(img_path))
        box = box_transformer(box, width, height)
        new_boxes.append(box)
    f.close()
    return new_boxes

def transform_anns(image_paths):
    all_boxes = []
    for img_path in image_paths:
        image = Image.open(img_path)
        width, height = image.size
        boxes = transform_group_anns(img_path, width, height)
        all_boxes.append(boxes)
    all_boxes = flatten(all_boxes)
    return all_boxes

def annotations_to_csv(annotation_list, csv_path):
    f = open(csv_path, 'w')
    for line in annotation_list:
        row = [str(_) for _ in line]
        row = ','.join(row) + '\n'
        f.write(row)
    f.close()

def train_test_split(image_paths, train_frac=0.7):
    num_train_samples = int(len(image_paths) * train_frac)
    train_paths = sample(image_paths, num_train_samples)
    test_paths = list(set(image_paths).difference(train_paths))
    return train_paths, test_paths

def make_class_map():
    class_map = [['p{}'.format(i), i] for i in range(num_distinct_classes)]
    return class_map

if __name__ == '__main__':
    raw_data_dir = Path(__file__).joinpath('..', 'raw_data').resolve()
    annotations_dir = Path(__file__).joinpath('..', 'annotations').resolve()
    image_paths = list(raw_data_dir.glob('*.jpg'))
    train_paths, test_paths = train_test_split(image_paths)
    train_paths, val_paths = train_test_split(train_paths, train_frac=0.8)
    class_map = make_class_map()
    unsplit_annotations = transform_anns(image_paths)
    train_annotations = transform_anns(train_paths)
    val_annotations = transform_anns(val_paths)
    test_annotations = transform_anns(test_paths)
    not_train_annotatinos = val_annotations + test_annotations

    annotations_to_csv(class_map, annotations_dir.joinpath('class_map.csv'))
    annotations_to_csv(not_train_annotatinos, annotations_dir.joinpath('not_train_annotatinos.csv'))
    annotations_to_csv(unsplit_annotations, annotations_dir.joinpath('unsplit.csv'))
    annotations_to_csv(train_annotations, annotations_dir.joinpath('train_annotations.csv'))
    annotations_to_csv(val_annotations, annotations_dir.joinpath('val_annotations.csv'))
    annotations_to_csv(test_annotations, annotations_dir.joinpath('test_annotations.csv'))
