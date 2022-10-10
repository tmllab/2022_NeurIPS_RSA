import os

import lmdb
import pyarrow as pa
import six
import torch.utils.data as data
from PIL import Image


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


class LmdbDataset(data.Dataset):

    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.transform = transform
        self.target_transform = target_transform

        env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path), readonly=True, lock=False, create=False, readahead=False, meminit=False)
        self.length = loads_pyarrow(env.begin(buffers=True).get(b'__len__'))
        print("Open Lmdb", self.db_path, self.length)
        env.close()

    def open_lmdb(self):
        self.env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path), readonly=True, lock=False, create=False, readahead=False, meminit=False)
        self.txn = self.env.begin(buffers=True)
        self.keys = loads_pyarrow(self.txn.get(b'__keys__'))

    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            self.open_lmdb()

        byteflow = self.txn.get(self.keys[index])

        imgbuf, target = loads_pyarrow(byteflow)
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # image transform
        if self.transform is not None:
            img = self.transform(img)

        # target transform
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + \
            '(db_path={},\n'.format(self.db_path) + \
            'transform={},\n'.format(self.transform) + \
            'target_transform={})'.format(self.target_transform)
