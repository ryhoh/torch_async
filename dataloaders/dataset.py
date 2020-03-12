from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
from os.path import join


class IMAGENET(Dataset):
    """ImageNet用オリジナルDataset

    ファイルパスカラム(path)とラベルカラム(label)を持つcsvを入力とし、
    Datasetをつくる

    Attributes:
        normalize (Normalize): 正規化?
        transform (transform): 前処理?
        _df (DataFrame): Dataset情報
        root_dir (str): データセットのルートディレクトリ
        images (dir): 画像の一覧

    Args:
        root_dir (str): データセットのルートディレクトリ
        csv_path (str): ファイルパスカラム(path)とラベルカラム(label)を持つcsv
        purpose (str): 目的フラグ。trainであればrandomcropとrandomflipを有効化

    Note:
        train時にDataSet内でrandomcropとrandomflipの前処理あり
    """
    def __init__(self, root_dir: str, csv_path: str, purpose: str):
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        # 前処理
        if purpose == "train":
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                self.normalize,
            ])
        self._df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        # csvから画像一覧を読み出す
        self.images = self._df['path'].values
        self.labels = self._df['label'].values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        image_name = self.images[idx]
        image = Image.open(join(self.root_dir, image_name))
        image = image.convert("RGB")
        if self.transforms:
            out_data = self.transforms(image)
        label = self.labels[idx]
        return out_data, int(label), image_name
