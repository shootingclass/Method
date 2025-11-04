import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from dataset_lstm import MethodDatasetLSTM


class MethodDataModuleLSTM(pl.LightningDataModule):
    def __init__(self, args, train_json, test_json, processed_root, class_to_idx,
                 batch_size=8, num_workers=8, val_ratio=0.1):
        super().__init__()
        self.args = args
        self.train_json = train_json
        self.test_json = test_json
        self.processed_root = processed_root
        self.class_to_idx = class_to_idx
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio

    def setup(self, stage=None):
        full_train = MethodDatasetLSTM(
            json_path=self.train_json,
            processed_root=self.processed_root,
            class_to_idx=self.class_to_idx,
        )

        n_total = len(full_train)
        n_val = max(1, int(n_total * self.val_ratio))
        n_train = n_total - n_val
        self.train_ds, self.val_ds = random_split(full_train, [n_train, n_val])

        self.test_ds = MethodDatasetLSTM(
            json_path=self.test_json,
            processed_root=self.processed_root,
            class_to_idx=self.class_to_idx,
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True,
                          collate_fn=MethodDatasetLSTM.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True,
                          collate_fn=MethodDatasetLSTM.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True,
                          collate_fn=MethodDatasetLSTM.collate_fn)
