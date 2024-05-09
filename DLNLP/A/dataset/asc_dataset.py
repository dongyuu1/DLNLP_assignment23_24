from torch.utils.data import Dataset
from .preprocess import load_data
from ..deberta.gpt2_tokenizer import GPT2Tokenizer
import os
import torch
from tqdm import tqdm


class DatasetASC(Dataset):
    def __init__(self, cfg, mode, sources=None):
        super(DatasetASC, self).__init__()
        if sources is None:
            sources = ["laptop", "rest"]
        self.cfg = cfg
        self.sources = sources
        self.mode = mode  # "train", "dev" or "test"
        data_paths = [os.path.join(self.cfg.DATA_ROOT_PATH, source) for source in self.sources]
        file_paths = [os.path.join(data_path, self.mode+".json") for data_path in data_paths]

        self.tokenizer = GPT2Tokenizer()
        data = self.preprocess_data(file_paths)

        self.token_ids = torch.tensor([item[0] for item in data], dtype=torch.long)
        self.segment_ids = torch.tensor([item[1] for item in data], dtype=torch.long)
        self.masks = torch.tensor([item[2] for item in data], dtype=torch.long)
        self.label_ids = torch.tensor([item[3] for item in data], dtype=torch.long)

    def preprocess_data(self, paths):
        label_map = {"positive": 0, "negative": 1, "neutral": 2}
        data_list = []
        for path in paths:
            data = load_data(path)
            for (i, id) in enumerate(data):
                token_list = []
                segment_list = []

                guid = str(id)
                label = data[id]['polarity']

                term = data[id]['term']
                term_tokens = self.tokenizer.tokenize(term)
                sentence = data[id]['sentence']
                sentence_tokens = self.tokenizer.tokenize(sentence)

                if len(term_tokens) + len(sentence_tokens) + 3 >= self.cfg.MAX_SEQ_LENGTH:
                    sentence_tokens = sentence_tokens[:self.cfg.MAX_SEQ_LENGTH - len(term_tokens) - 3]

                token_list.append("[CLS]")
                segment_list.append(0)

                for token in sentence_tokens:
                    token_list.append(token)
                    segment_list.append(0)

                token_list.append("[SEP]")
                segment_list.append(0)

                for token in term_tokens:
                    token_list.append(token)
                    segment_list.append(1)
                token_list.append("[SEP]")
                segment_list.append(1)

                token_id_list = self.tokenizer.convert_tokens_to_ids(token_list)
                mask_list = [1] * len(token_id_list)

                while len(token_id_list) < self.cfg.MAX_SEQ_LENGTH:
                    token_id_list.append(0)
                    segment_list.append(0)
                    mask_list.append(0)

                label_id = label_map[label]

                data_list.append((token_id_list, segment_list, mask_list, label_id))

        return data_list

    def __getitem__(self, ind):
        return self.token_ids[ind], self.segment_ids[ind], self.masks[ind], self.label_ids[ind]

    def __len__(self):
        return self.token_ids.shape[0]
