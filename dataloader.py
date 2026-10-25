import torch
import torch.utils.data as data
import operator
from collections import Counter


class PoisonedDataset(data.Dataset):
    def __init__(self, path, subset='clean_train', transform=None, num_per_label=-1, poison_tuple_list=None,
                 poison_indices=None, class_labels=None, subset_group=0, preserve_order=False):
        """
        Replace bases with clean-label poisons within one contiguous class window.

        Positive num_per_label is the total per-class budget, including poisons.
        Nonpositive budgets select all allowed samples while still replacing bases.
        """
        # Project splits contain PIL images; load only trusted local artifacts.
        self.img_label_list = torch.load(path, map_location='cpu', weights_only=False)[subset]
        self.transform = transform
        self.preserve_order = preserve_order
        self.poison_indices = list(poison_indices) if poison_indices is not None else []
        self.poison_tuple_list = list(poison_tuple_list) if poison_tuple_list is not None else []
        class_labels = list(range(10)) if class_labels is None else list(class_labels)
        self.get_valid_indices(num_per_label, self.poison_indices, class_labels, subset_group)

    def get_valid_indices(self, num_per_label, poison_indices, class_labels, subset_group):
        try:
            num_per_label = operator.index(num_per_label)
            subset_group = operator.index(subset_group)
            class_labels = [operator.index(label) for label in class_labels]
            poison_indices = [operator.index(index) for index in poison_indices]
        except TypeError as error:
            raise ValueError('Budgets, group, labels and base indices must be integers') from error
        if subset_group < 0:
            raise ValueError('subset_group must be nonnegative')
        if num_per_label <= 0 and subset_group != 0:
            raise ValueError('subset_group requires a positive num_per_label budget')
        if not class_labels or len(set(class_labels)) != len(class_labels):
            raise ValueError('class_labels must be nonempty and unique')
        if len(poison_indices) != len(self.poison_tuple_list):
            raise ValueError('Each poison must have exactly one base index')
        if len(set(poison_indices)) != len(poison_indices):
            raise ValueError('Poison base indices must be unique')
        allowed_labels = set(class_labels)
        poison_counts = Counter()
        for poison, index in zip(self.poison_tuple_list, poison_indices):
            if index < 0 or index >= len(self.img_label_list):
                raise ValueError(f'Poison base index {index} is outside the selected split')
            if not isinstance(poison, (tuple, list)) or len(poison) != 2:
                raise ValueError('Each poison must be an (image, label) pair')
            _, label = poison
            try:
                label = operator.index(label)
            except TypeError as error:
                raise ValueError('Poison labels must be integers') from error
            base_label = int(self.img_label_list[index][1])
            if label != base_label:
                raise ValueError(f'Clean-label poison {index} has label {label}; base has {base_label}')
            if label not in allowed_labels:
                raise ValueError(f'Poison class {label} is absent from class_labels')
            poison_counts[label] += 1
        if num_per_label > 0 and any(count > num_per_label for count in poison_counts.values()):
            raise ValueError('Poison count exceeds the per-class training budget')

        self.poison_indices = poison_indices
        removed = set(poison_indices)
        cursors = Counter()
        selected = set()
        self.valid_indices = []
        start = subset_group * num_per_label if num_per_label > 0 else 0
        stop = start + num_per_label if num_per_label > 0 else None
        for index, (_, label) in enumerate(self.img_label_list):
            label = int(label)
            if label not in allowed_labels:
                continue
            ordinal = cursors[label]
            cursors[label] += 1
            if ordinal >= start and (stop is None or ordinal < stop):
                selected.add(index)
                if index not in removed:
                    self.valid_indices.append(index)
        if stop is not None:
            shortages = {label: cursors[label] for label in class_labels if cursors[label] < stop}
            if shortages:
                raise ValueError(f'Class window [{start}, {stop}) exceeds available samples: {shortages}')
        outside = removed - selected
        if outside:
            raise ValueError(f'Poison bases must belong to the selected class window: {sorted(outside)}')
        self.class_counts = dict(Counter(int(self.img_label_list[i][1]) for i in self.valid_indices)
                                 + poison_counts)
        self.ordered_indices = sorted(selected)
        self.replacements = dict(zip(poison_indices, self.poison_tuple_list))

    def __len__(self):
        return len(self.valid_indices) + len(self.poison_tuple_list)

    def __getitem__(self, index):
        if self.preserve_order:
            source_index = self.ordered_indices[index]
            if source_index in self.replacements:
                return self.replacements[source_index]
            img, label = self.img_label_list[source_index]
            return (self.transform(img) if self.transform is not None else img), label
        if index < len(self.poison_tuple_list):
            img, label = self.poison_tuple_list[index]
        else:
            idx = self.valid_indices[index - len(self.poison_tuple_list)]
            img, label = self.img_label_list[idx]
            if self.transform is not None:
                img = self.transform(img)

        return img, label


class FeatureSet(data.Dataset):
    """Cache frozen penultimate features from a model or a DataParallel model."""

    def __init__(self, train_loader, net, device):
        device = torch.device(device)
        model = net.module if isinstance(net, torch.nn.DataParallel) else net
        feat_list, label_list = [], []
        was_training = net.training
        net.eval()
        try:
            with torch.no_grad():
                for input, target in train_loader:
                    input, target = input.to(device), target.to(device)
                    feat_list.append(model.penultimate(input).detach())
                    label_list.append(target)
        finally:
            net.train(was_training)
        if not feat_list:
            raise ValueError('Cannot extract features from an empty training loader')
        self.feature_tensor = torch.cat(feat_list, 0)
        self.label_tensor = torch.cat(label_list, 0)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        feature, target = self.feature_tensor[index], self.label_tensor[index]

        return feature, target

    def __len__(self):
        return self.feature_tensor.size(0)


class SubsetOfList(data.Dataset):
    def __init__(self, img_label_list, transform=None, start_idx=0, end_idx=1e10,
                    poison_tuple_list=[],
                    class_labels=[i for i in range(10)]):
        # logistics work for mixing data from CINIC10 with CIFAR10
        # suppose the poison samples are disjoint with the CINIC dataset
        self.img_label_list = img_label_list #torch.load(path)[subset]
        self.transform = transform
        self.poison_tuple_list = poison_tuple_list
        self.get_valid_list(start_idx, end_idx, class_labels)

    def get_valid_list(self, start_idx, end_idx, class_labels):
        # remove poisoned ones
        num_per_label_dict = {}
        selected_img_label_list = [] #[pt for pt in self.poison_tuple_list]
        if len(self.poison_tuple_list) > 0:
            poison_label = self.poison_tuple_list[0][1]
            print("Poison label: {}".format(poison_label))
        else:
            poison_label = -1

        for idx, (img, label) in enumerate(self.img_label_list):
            if label not in class_labels:
                continue
            if label not in num_per_label_dict:
                num_per_label_dict[label] = 0
            if num_per_label_dict[label] >= start_idx and num_per_label_dict[label] < end_idx:
                if label == poison_label and num_per_label_dict[label] - start_idx < len(self.poison_tuple_list):
                    pass
                else:
                    selected_img_label_list.append([img, label])
            num_per_label_dict[label] += 1

        self.img_label_list = selected_img_label_list


    def __len__(self):
        return len(self.img_label_list) + len(self.poison_tuple_list)

    def __getitem__(self, index):
        if index < len(self.poison_tuple_list):
            img, label = self.poison_tuple_list[index]
        else:
            img, label = self.img_label_list[index-len(self.poison_tuple_list)]
            if self.transform is not None:
                img = self.transform(img)

        return img, label
