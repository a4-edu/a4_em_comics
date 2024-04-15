import csv
from functools import cached_property 
import json
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np


class EmComDataItem:
    """
    Emotion Recoginition Comics Data Item
    """
    LABEL_NAMES = (
        "angry", #0
        "disgust", #1
        "fear", #2
        "happy", #3
        "sad", #4
        "surprise", #5
        "neutral", #6
        "other", #7
    )
    
    def __init__(self, image_path: Path, dialog_text: list[str], 
                 narration_text: list[str], label_ids: list[int] | None = None):
        assert image_path.exists(), image_path
        if label_ids is not None:
            assert all(0<=x<len(self.LABEL_NAMES) for x in label_ids), label_ids
        
        self.image_path = image_path
        self.dialog_text = dialog_text
        self.narration_text = narration_text
        self.label_ids = label_ids
    
    def __repr__(self) -> str:
        return f"EmComDataItem(image_path={self.image_path},"\
            " dialog_text={self.dialog_text[:10]}({len(self.dialog_text)}), labels={self.labels})"
    
    @cached_property
    def image(self) -> np.ndarray:
        return plt.imread(str(self.image_path)).astype(np.uint32)
    
    @property
    def text(self) -> list[str]:
        """
        Both narration and dialog texts
        """
        return self.dialog_text + self.narration_text

    @property
    def labels(self) -> list[str] | None:
        """
        Labels as a sequence of strings
        """
        if not self.label_ids:
            return
        return [
            self.LABEL_NAMES[idx]
            for idx in self.label_ids
        ]
    

class EmComDataSet:
    """
    Emotion Recognition Comics Dataset
    """
    @staticmethod
    def read_labels(labels_file: Path) -> dict[str, list[int]]:
        assert labels_file.exists(), labels_file
        labels_data = {}
        with open(labels_file, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                labels_data[row['image_id']] = []

                for emotion_idx, emotion in enumerate(EmComDataItem.LABEL_NAMES):
                    assert emotion in row.keys(), (emotion, row.keys())
                    if int(row[emotion]):
                        labels_data[row['image_id']].append(emotion_idx)
        return labels_data

    @classmethod
    def read_from_root(cls, root_path: Path, *, val_subset: bool = False) -> 'EmComDataSet':
        assert root_path.exists(), root_path
        suffix = 'val' if val_subset else 'train'
        images_dir = root_path / "images"
        transcriptions_file = root_path / f'input_{suffix}.json'
        assert images_dir.exists(), images_dir
        assert transcriptions_file.exists(), transcriptions_file
        
        labels_file = root_path / f'labels.csv'
        labels_data = cls.read_labels(labels_file)
        
        image_names = set([x.stem for x in sorted(images_dir.glob('*.jpg'))])
        items = []

        with open(transcriptions_file, 'r') as f:
            transcriptions = json.loads(f.read())
            for record in transcriptions:
                assert record["img_id"] in image_names, (record["img_id"], list(image_names)[:2])
                items.append(
                    EmComDataItem(
                        image_path=images_dir/f'{record["img_id"]}.jpg',
                        narration_text=record["narration"],
                        dialog_text=record["dialog"],
                        label_ids=labels_data[record["img_id"]]
                    )
                )
        return cls(items)

    def __init__(self, items: list[EmComDataItem]):
        self.items = items
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __repr__(self) -> str:
        return f"DataSet(items {len(self.items)})"
    
    def __getitem__(self, index: int) -> EmComDataItem:
        return self.items[index]

    def as_label_probs(self) -> dict[str, tuple[int]]:
        """
        Convert dataset labels into 'probability predictions' format, i.e.
        dict with image names as keys and probability for every emotion as values.
        Example:
        {
            "<image_0>.jpg": (0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0), 
            "<image_1>.jpg": (1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ...
        }
        """
        def sparse_to_dense(sparse_ids):
            return tuple(
                1.0 if label_idx in sparse_ids else 0.0
                for label_idx in range(len(EmComDataItem.LABEL_NAMES))
            )
            
        return dict(
            (it.image_path.name, sparse_to_dense(it.label_ids))
            for it in self.items
        )
