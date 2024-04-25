from collections import OrderedDict
import numpy as np
import torch
from torchvision.datasets.mnist import MNIST
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageDraw, ImageOps
import json
import os
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModel
import os
import argparse
import torch
import cv2

class Padding(object):
    def __init__(self, max_length, vocab_size):
        self.max_length = max_length
        self.bos_token = vocab_size - 3
        self.eos_token = vocab_size - 2
        self.pad_token = vocab_size - 1

    def __call__(self, layout):
        # grab a chunk of (max_length + 1) from the layout

        # layout = layout[:self.max_length-1]
        chunk = torch.zeros(self.max_length+1, dtype=torch.long) + self.pad_token
        # Assume len(item) will always be <= self.max_length:
        chunk[0] = self.bos_token
        chunk[1:len(layout)+1] = layout
        chunk[len(layout)+1] = self.eos_token

        x = chunk[:-1]
        y = chunk[1:]
        return {'x': x, 'y': y}


class JSONLayout(Dataset):
    def __init__(self, json_path, max_length=None, precision=8):
        with open(json_path, "r") as f:
            data = json.loads(f.read())

        self.root_dir = os.path.dirname(json_path)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        # self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        images, annotations, categories = data['images'], data['annotations'], data['categories']
        # self.size = pow(2, precision)
        self.size = 224

        self.categories = {c["id"]: c for c in categories}
        self.colors = gen_colors(len(self.categories))

        self.json_category_id_to_contiguous_id = {
            v: i + self.size for i, v in enumerate([c["id"] for c in self.categories.values()])
        }

        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        self.vocab_size = self.size + len(self.categories) + 3  # bos, eos, pad tokens
        self.bos_token = self.vocab_size - 3
        self.eos_token = self.vocab_size - 2
        self.pad_token = self.vocab_size - 1

        image_to_annotations = {}
        for annotation in annotations:
            image_id = annotation["image_id"]

            if not (image_id in image_to_annotations):
                image_to_annotations[image_id] = []

            image_to_annotations[image_id].append(annotation)

        self.data = [] #(image, layout, prompt)
        for image in images:
            image_id = image["id"]
            height, width = float(image["height"]), float(image["width"])
            # height, width = 224.0, 224.0
            
            #resize image to 224
            image_array = np.array(Image.open(os.path.join(self.root_dir, image["file_name"])).resize((224, 224)))
            # image_array = np.array(Image.open(os.path.join(self.root_dir, image["file_name"])))

            if image_id not in image_to_annotations:
                continue

            ann_box = []
            ann_cat = []
            defects = []
            for ann in image_to_annotations[image_id]:
                # for i in ann['segments_info']:
                x, y, w, h = ann["bbox"]
                ann_box.append([x, y, w, h])
                ann_cat.append(self.json_category_id_to_contiguous_id[ann["category_id"]])
                defects.append(self.categories[ann['category_id']]['name'])
            
            prompt = "An image of printed circuit board"
            if len(defects) != 0:
                prompt = prompt + " with " + ', '.join(defects) + " defects in it"

            #tokenize
            token = self.tokenizer(prompt, truncation=True, max_length=50, padding="max_length", return_tensors="pt")
            # text = self.text_encoder(**token).last_hidden_state
            # Sort boxes
            if len(ann_box) == 0:
                continue
            ann_box = np.array(ann_box, dtype=np.float32)
            ind = np.lexsort((ann_box[:, 0], ann_box[:, 1]))
            ann_box = ann_box[ind]
       
            ann_cat = np.array(ann_cat)
            ann_cat = ann_cat[ind]

            # Discretize boxes
            ann_box = self.quantize_box(ann_box, width, height)

            # Append the categories
            layout = np.concatenate([ann_cat.reshape(-1, 1), ann_box], axis=1)

            if len(layout.reshape(-1)) + 2 >= max_length:
                continue 

            # Flatten and add to the dataset
            self.data.append((image_array, token, layout.reshape(-1)))

        self.max_length = max_length
        if self.max_length is None:
            self.max_length = max([len(x[2]) for x in self.data]) + 2  # bos, eos tokens
        self.transform = Padding(self.max_length, self.vocab_size)

    def quantize_box(self, boxes, width, height):

        # range of xy is [0, large_side-1]
        # range of wh is [1, large_side]
        # bring xywh to [0, 1]
        boxes[:, [2, 3]] = boxes[:, [2, 3]] - 1
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / (width - 1.0)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / (height - 1.0)
        boxes = np.clip(boxes, 0, 1)

        # next take xywh to [0, size-1]
        boxes = (boxes * (self.size - 1)).round()

        return boxes.astype(np.int32)

    def __len__(self):
        return len(self.data)

    def render(self, layout, background):
        background = np.array(background.permute(1, 2, 0), dtype=np.uint8)
        #to RGB image
        # img = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        #to Image format
        img = Image.fromarray(background)
        # img = Image.new('RGB', (224, 224), color=(255, 255, 255))
        draw = ImageDraw.Draw(img, 'RGBA')
        layout = layout.reshape(-1)
        layout = trim_tokens(layout, self.bos_token, self.eos_token, self.pad_token)
        layout = layout[: len(layout) // 5 * 5].reshape(-1, 5)
        box = layout[:, 1:].astype(np.float32)                             # (-1, 4)
        box[:, [0, 1]] = box[:, [0, 1]] / (self.size - 1) * 223            # left corner coordinates to [0, 224)
        box[:, [2, 3]] = box[:, [2, 3]] / self.size * 224                  # width and height to [0, 224] 
        box[:, [2, 3]] = box[:, [0, 1]] + box[:, [2, 3]]                   # right corner coordinates to [0, 224] 

        for i in range(len(layout)):
            x1, y1, x2, y2 = box[i]
            cat = layout[i][0]
            col = self.colors[cat-self.size] if 0 <= cat-self.size < len(self.colors) else [0, 0, 0]
            draw.rectangle([x1, y1, x2, y2],
                           outline=tuple(col) + (200,),
                           fill=tuple(col) + (64,),
                           width=2)

        # Add border around image
        img = ImageOps.expand(img, border=2)
        return img

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) tokens from the data
        image, token, layout = self.data[idx]
        layout = torch.tensor(layout, dtype=torch.long)
        image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)
        layout = self.transform(layout)
        # return layout['x'], layout['y'], image, token['input_ids'], token['attention_mask']
        return layout['x'], layout['y'], image, token


class COCOLayout(Dataset):
    def __init__(self, json_path, max_length=None, precision=8):
        with open(json_path, "r") as f:
            data = json.loads(f.read())

        images, annotations, categories = data['images'], data['annotations'], data['categories']
        # self.size = pow(2, precision)
        self.size = 224

        self.categories = {c["id"]: c for c in categories}
        self.colors = gen_colors(len(self.categories))

        self.json_category_id_to_contiguous_id = {
            v: i + self.size for i, v in enumerate([c["id"] for c in self.categories.values()])
        }

        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        self.vocab_size = self.size + len(self.categories) + 3  # bos, eos, pad tokens
        self.bos_token = self.vocab_size - 3
        self.eos_token = self.vocab_size - 2
        self.pad_token = self.vocab_size - 1

        image_to_annotations = {}
        for annotation in annotations:
            image_id = annotation["image_id"]

            if not (image_id in image_to_annotations):
                image_to_annotations[image_id] = []

            image_to_annotations[image_id].append(annotation)

        self.data = []
        for image in images:
            image_id = image["id"]
            height, width = float(image["height"]), float(image["width"])

            if image_id not in image_to_annotations:
                continue

            ann_box = []
            ann_cat = []
            for ann in image_to_annotations[image_id]:
                for i in ann['segments_info']:
                    x, y, w, h = i["bbox"]
                    ann_box.append([x, y, w, h])
                    ann_cat.append(self.json_category_id_to_contiguous_id[i["category_id"]])
            ann_box = ann_box[:9]
            ann_cat = ann_cat[:9]

            # Sort boxes
            if len(ann_box) == 0:
                continue
            ann_box = np.array(ann_box, dtype=np.float32)
            ind = np.lexsort((ann_box[:, 0], ann_box[:, 1]))
            ann_box = ann_box[ind]
       
            ann_cat = np.array(ann_cat)
            ann_cat = ann_cat[ind]

            # Discretize boxes
            ann_box = self.quantize_box(ann_box, width, height)

            # Append the categories
            layout = np.concatenate([ann_cat.reshape(-1, 1), ann_box], axis=1)

            # Flatten and add to the dataset
            self.data.append(layout.reshape(-1))

        self.max_length = max_length
        if self.max_length is None:
            self.max_length = max([len(x) for x in self.data]) + 2  # bos, eos tokens
        self.transform = Padding(self.max_length, self.vocab_size)

    def quantize_box(self, boxes, width, height):

        # range of xy is [0, large_side-1]
        # range of wh is [1, large_side]
        # bring xywh to [0, 1]
        boxes[:, [2, 3]] = boxes[:, [2, 3]] - 1
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / (width - 1.0)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / (height - 1.0)
        boxes = np.clip(boxes, 0, 1)

        # next take xywh to [0, size-1]
        boxes = (boxes * (self.size - 1)).round()

        return boxes.astype(np.int32)

    def __len__(self):
        return len(self.data)

    def render(self, layout):
        img = Image.new('RGB', (224, 224), color=(255, 255, 255))
        draw = ImageDraw.Draw(img, 'RGBA')
        layout = layout.reshape(-1)
        layout = trim_tokens(layout, self.bos_token, self.eos_token, self.pad_token)
        layout = layout[: len(layout) // 5 * 5].reshape(-1, 5)
        box = layout[:, 1:].astype(np.float32)
        box[:, [0, 1]] = box[:, [0, 1]] / (self.size - 1) * 223            # left corner coordinates to [0, 224)
        box[:, [2, 3]] = box[:, [2, 3]] / self.size * 224 
        box[:, [2, 3]] = box[:, [0, 1]] + box[:, [2, 3]]

        for i in range(len(layout)):
            x1, y1, x2, y2 = box[i]
            cat = layout[i][0]
            col = self.colors[cat-self.size] if 0 <= cat-self.size < len(self.colors) else [0, 0, 0]
            draw.rectangle([x1, y1, x2, y2],
                           outline=tuple(col) + (200,),
                           fill=tuple(col) + (64,),
                           width=2)

        # Add border around image
        img = ImageOps.expand(img, border=2)
        return img

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) tokens from the data
        layout = torch.tensor(self.data[idx], dtype=torch.long)
        layout = self.transform(layout)
        return layout['x'], layout['y']

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        
        for k,v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
    
class CausalTextCrossAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(512, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(512, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, z, layer_past=None):
        B, T, C = x.size()
        _, T_, _ = z.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(z).view(B, T_, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(z).view(B, T_, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T_) -> (B, nh, T, T_)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att[:,:,:T,:T] = att[:,:,:T,:T].masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T_) x (B, nh, T_, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class CausalImageCrossAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(768, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(768, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, z, layer_past=None):
        # B, T, C = x.size()

        # # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # k = self.key(z).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # v = self.value(z).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # att = self.attn_drop(att)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # # output projection
        # y = self.resid_drop(self.proj(y))
        # return y
        B, T, C = x.size()
        _, T_, _ = z.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(z).view(B, T_, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(z).view(B, T_, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T_) -> (B, nh, T, T_)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att[:,:,:T,:T] = att[:,:,:T,:T].masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T_) x (B, nh, T_, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.ln4 = nn.LayerNorm(config.n_embd)
        # self.ln5 = nn.LayerNorm(512)
        # self.ln6 = nn.LayerNorm(768)
        
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.crosstextattn = CausalTextCrossAttention(config)
        self.crossimageattn = CausalImageCrossAttention(config)
        self.image_available = config.image_available
        self.text_available = config.text_available

    def forward(self, x, text_last_hidden_state, image_last_hidden_state):
        x = x + self.attn(self.ln1(x))
        if self.text_available and text_last_hidden_state is not None:
            x = x + self.crosstextattn(self.ln3(x), text_last_hidden_state)
        if self.image_available:
            x = x + self.crossimageattn(self.ln4(x), image_last_hidden_state)
        # x = x + self.crosstextattn(self.ln3(x), text_last_hidden_state)
        # x = x + self.crossimageattn(self.ln4(x), image_last_hidden_state)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, text_last_hidden_state, image_last_hidden_state, targets=None, pad_token=-100):
        b, t = idx.size() 
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector

        
        x = self.drop(token_embeddings + position_embeddings)
        for block in self.blocks:
            x = block(x, text_last_hidden_state, image_last_hidden_state) 
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=pad_token)

        return logits, loss

import os
import math
import logging
import wandb

from tqdm import tqdm
import numpy as np

import torch
from torch.nn import functional as F

from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_iters = 0
    final_iters = 0  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_dir = None
    samples_dir = None
    sample_every = 1
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, args):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.iters = 0
        self.fixed_x = None
        self.fixed_y = None
        print("Using wandb")
        wandb.login(key="139751fd25a8a79f7458b6410f64d0a1a9d0a3f3")
        wandb.init(project=args.project_name, name=args.exp)
        wandb.config.update(args)

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.sample_with_text = args.sample_with_text
    
    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        ckpt_path = os.path.join(self.config.ckpt_dir, 'checkpoint.pth')
        logger.info("saving %s", ckpt_path)
        torch.save(raw_model.state_dict(), ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        pad_token = self.train_dataset.vocab_size - 1
        
        # load pretrained weights
        if config.pretrained_ckpt is not None:
            model_state_dict = torch.load(self.config.pretrained_ckpt, map_location='cpu')
            state_dict = OrderedDict()
            for k, v in model_state_dict.items():
                if k.startswith('tok_emb') or k.startswith('head'):
                    continue
                k = 'module.'+ k
                state_dict[k] = v
            re = model.load_state_dict(state_dict, strict=False)
            print("Successfully load pretrained model from ", self.config.pretrained_ckpt, re)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y, image, token) in pbar:

                # import ipdb; ipdb.set_trace()
                for k, v in token.items():
                    token[k] = v.squeeze(1)
                    
                text_last_hidden_state = self.text_encoder(**token).last_hidden_state
                image_last_hidden_state = self.image_encoder(image).last_hidden_state
                
                if not is_train:
                    self.fixed_x = x[:min(4, len(x))]
                    self.fixed_y = y[:min(4, len(y))]
                    self.fixed_text_last_hidden_state = text_last_hidden_state[:min(4, len(text_last_hidden_state))]
                    self.fixed_image_last_hidden_state = image_last_hidden_state[:min(4, len(image_last_hidden_state))]
                    self.fixed_image = image[:min(4, len(image))]
                    # self.fixed_token = token['input_ids'][:min(4, len(token['input_ids']))]

                # break

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    # import ipdb; ipdb.set_trace()
                    logits, loss = model(x, text_last_hidden_state, image_last_hidden_state, y, pad_token=pad_token)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()
                    self.iters += 1
                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        # self.tokens += (y >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.iters < config.warmup_iters:
                            # linear warmup
                            lr_mult = float(self.iters) / float(max(1, config.warmup_iters))
                        else:
                            # cosine learning rate decay
                            progress = float(self.iters - config.warmup_iters) / float(max(1, config.final_iters - config.warmup_iters))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    wandb.log({
                        'train loss': loss.item(),
                        'lr': lr, 'epoch': epoch+1
                    }, step=self.iters)
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                wandb.log({'test loss': test_loss}, step=self.iters)
                return test_loss

        best_loss = float('inf')

        for epoch in range(config.max_epochs):
            run_epoch('train')
            if self.test_dataset is not None:
                with torch.no_grad():
                    test_loss = run_epoch('test')

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_dir is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint()

            # sample from the model
            if self.config.samples_dir is not None and (epoch+1) % self.config.sample_every == 0:
                # import ipdb; ipdb.set_trace()
                # inputs
                layouts = self.fixed_x.detach().cpu().numpy()
                input_layouts = [self.train_dataset.render(layout, image) for layout, image in zip(layouts, self.fixed_image)]
                # for i, layout in enumerate(layouts):
                #     layout = self.train_dataset.render(layout)
                #     layout.save(os.path.join(self.config.samples_dir, f'input_{epoch:02d}_{i:02d}.png'))

                # reconstruction
                x_cond = self.fixed_x.to(self.device)
                logits, _ = model(x_cond, self.fixed_text_last_hidden_state, self.fixed_image_last_hidden_state)
                probs = F.softmax(logits, dim=-1)
                _, y = torch.topk(probs, k=1, dim=-1)
                layouts = torch.cat((x_cond[:, :6], y[:, :, 0]), dim=1).detach().cpu().numpy()
                recon_layouts = [self.train_dataset.render(layout, image) for layout, image in zip(layouts, self.fixed_image)]
                # for i, layout in enumerate(layouts):
                #     layout = self.train_dataset.render(layout)
                #     layout.save(os.path.join(self.config.samples_dir, f'recon_{epoch:02d}_{i:02d}.png'))
                
                # samples - fixed x
                if self.sample_with_text:
                    layouts = sample(model, x_cond[:, :2], self.fixed_text_last_hidden_state, self.fixed_image_last_hidden_state, steps=self.train_dataset.max_length,
                                 temperature=1.0, sample=False, top_k=None, real_x=x_cond).detach().cpu().numpy()
                    layouts_fixed = sample_without_real(model, x_cond[:, :2], self.fixed_text_last_hidden_state, self.fixed_image_last_hidden_state, steps=self.train_dataset.max_length,
                                 temperature=1.0, sample=False, top_k=None, real_x=x_cond).detach().cpu().numpy()
                else:
                    layouts = sample(model, x_cond[:, :2], None, self.fixed_image_last_hidden_state, steps=self.train_dataset.max_length,
                                 temperature=1.0, sample=False, top_k=None, real_x=x_cond).detach().cpu().numpy()
                print(layouts)
                sample_cond_layouts = [self.train_dataset.render(layout, image) for layout, image in zip(layouts, self.fixed_image)]
                sample_cond_layouts_fixed = [self.train_dataset.render(layout, image) for layout, image in zip(layouts_fixed, self.fixed_image)]
                # for i, layout in enumerate(layouts):
                #     layout = self.train_dataset.render(layout)
                #     layout.save(os.path.join(self.config.samples_dir, f'sample_det_{epoch:02d}_{i:02d}.png'))

                # samples - random
                layouts = sample_without_real(model, x_cond[:, :2], self.fixed_text_last_hidden_state, self.fixed_image_last_hidden_state, steps=self.train_dataset.max_length,
                                 temperature=1.0, sample=True, top_k=5).detach().cpu().numpy()
                sample_random_layouts = [self.train_dataset.render(layout, image) for layout, image in zip(layouts, self.fixed_image)]
                # for i, layout in enumerate(layouts):
                #     layout = self.train_dataset.render(layout)
                #     layout.save(os.path.join(self.config.samples_dir, f'sample_random_{epoch:02d}_{i:02d}.png'))

                # samples - deterministic
                layouts = sample_without_real(model, x_cond[:, :2], self.fixed_text_last_hidden_state, self.fixed_image_last_hidden_state, steps=self.train_dataset.max_length,
                                 temperature=1.0, sample=False, top_k=None).detach().cpu().numpy()
                sample_det_layouts = [self.train_dataset.render(layout, image) for layout, image in zip(layouts, self.fixed_image)]
                # for i, layout in enumerate(layouts):
                #     layout = self.train_dataset.render(layout)
                #     layout.save(os.path.join(self.config.samples_dir, f'sample_det_{epoch:02d}_{i:02d}.png'))
                
                defects = []
                for i in self.fixed_x:
                    caption = []
                    for j in range(0,len(i[1:(i == 234).nonzero(as_tuple=True)[0][0]]),5):
                        caption.append(train_dataset.categories[i[1:(i == 234).nonzero(as_tuple=True)[0][0]][j].item()-224]['name'])
                    caption = ','.join(caption)
                    defects.append(caption)
                    
                wandb.log({
                    "input_layouts": [wandb.Image(pil, caption=f'input_{defects[i]}.png')
                                      for i, pil, in enumerate(input_layouts)],
                    "recon_layouts": [wandb.Image(pil, caption=f'recon_{defects[i]}.png')
                                      for i, pil, in enumerate(recon_layouts)],
                    "sample_random_layouts": [wandb.Image(pil, caption=f'random_{defects[i]}.png')
                                      for i, pil, in enumerate(sample_random_layouts)],
                    "sample_det_layouts": [wandb.Image(pil, caption=f'det_{defects[i]}.png')
                                      for i, pil, in enumerate(sample_det_layouts)],
                    "sample_cond_layouts": [wandb.Image(pil, caption=f'cond_{defects[i]}.png')
                                      for i, pil, in enumerate(sample_cond_layouts)],
                    "sample_cond_layouts_fixed": [wandb.Image(pil, caption=f'cond_fixed_{defects[i]}.png')
                                      for i, pil, in enumerate(sample_cond_layouts_fixed)],
                }, step=self.iters)
    
    def pretrain(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        pad_token = self.train_dataset.vocab_size - 1
        
        # load pretrained weights
        if config.pretrained_ckpt is not None:
            model_state_dict = torch.load(self.config.pretrained_ckpt, map_location='cpu')
            model.load_state_dict(model_state_dict, strict=False)
            print("Successfully load pretrained model from ", self.config.pretrained_ckpt)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y) in pbar:

                # import ipdb; ipdb.set_trace()
                # for k, v in token.items():
                #     token[k] = v.squeeze(1)
                    
                # text_last_hidden_state = self.text_encoder(**token).last_hidden_state
                # image_last_hidden_state = self.image_encoder(image).last_hidden_state
                
                if not is_train:
                    self.fixed_x = x[:min(4, len(x))]
                    self.fixed_y = y[:min(4, len(y))]
                    self.fixed_text_last_hidden_state = None
                    self.fixed_image_last_hidden_state = None
                    # self.fixed_image = image[:min(4, len(image))]
                    # self.fixed_token = token['input_ids'][:min(4, len(token['input_ids']))]

                # break

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    # import ipdb; ipdb.set_trace()
                    logits, loss = model(x, None, None, y, pad_token=pad_token)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()
                    self.iters += 1
                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        # self.tokens += (y >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.iters < config.warmup_iters:
                            # linear warmup
                            lr_mult = float(self.iters) / float(max(1, config.warmup_iters))
                        else:
                            # cosine learning rate decay
                            progress = float(self.iters - config.warmup_iters) / float(max(1, config.final_iters - config.warmup_iters))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    wandb.log({
                        'train loss': loss.item(),
                        'lr': lr, 'epoch': epoch+1
                    }, step=self.iters)
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                wandb.log({'test loss': test_loss}, step=self.iters)
                return test_loss

        best_loss = float('inf')

        for epoch in range(config.max_epochs):
            run_epoch('train')
            if self.test_dataset is not None:
                with torch.no_grad():
                    test_loss = run_epoch('test')

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_dir is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint()

            # sample from the model
            if self.config.samples_dir is not None and (epoch+1) % self.config.sample_every == 0:
                # import ipdb; ipdb.set_trace()
                # inputs
                layouts = self.fixed_x.detach().cpu().numpy()
                input_layouts = [self.train_dataset.render(layout) for layout in layouts]
                # for i, layout in enumerate(layouts):
                #     layout = self.train_dataset.render(layout)
                #     layout.save(os.path.join(self.config.samples_dir, f'input_{epoch:02d}_{i:02d}.png'))

                # reconstruction
                x_cond = self.fixed_x.to(self.device)
                logits, _ = model(x_cond, self.fixed_text_last_hidden_state, self.fixed_image_last_hidden_state)
                probs = F.softmax(logits, dim=-1)
                _, y = torch.topk(probs, k=1, dim=-1)
                layouts = torch.cat((x_cond[:, :6], y[:, :, 0]), dim=1).detach().cpu().numpy()
                recon_layouts = [self.train_dataset.render(layout) for layout in layouts]
                # for i, layout in enumerate(layouts):
                #     layout = self.train_dataset.render(layout)
                #     layout.save(os.path.join(self.config.samples_dir, f'recon_{epoch:02d}_{i:02d}.png'))
                
                # samples - fixed x
                if self.sample_with_text:
                    layouts = sample(model, x_cond[:, :2], self.fixed_text_last_hidden_state, self.fixed_image_last_hidden_state, steps=self.train_dataset.max_length,
                                 temperature=1.0, sample=False, top_k=None, real_x=x_cond).detach().cpu().numpy()
                else:
                    layouts = sample(model, x_cond[:, :6], None, self.fixed_image_last_hidden_state, steps=self.train_dataset.max_length,
                                 temperature=1.0, sample=False, top_k=None, real_x=x_cond).detach().cpu().numpy()
                print(layouts)
                sample_cond_layouts = [self.train_dataset.render(layout) for layout in layouts]
                # for i, layout in enumerate(layouts):
                #     layout = self.train_dataset.render(layout)
                #     layout.save(os.path.join(self.config.samples_dir, f'sample_det_{epoch:02d}_{i:02d}.png'))

                # samples - random
                layouts = sample_without_real(model, x_cond[:, :2], self.fixed_text_last_hidden_state, self.fixed_image_last_hidden_state, steps=self.train_dataset.max_length,
                                 temperature=1.0, sample=True, top_k=5).detach().cpu().numpy()
                sample_random_layouts = [self.train_dataset.render(layout) for layout in layouts]
                # for i, layout in enumerate(layouts):
                #     layout = self.train_dataset.render(layout)
                #     layout.save(os.path.join(self.config.samples_dir, f'sample_random_{epoch:02d}_{i:02d}.png'))

                # samples - deterministic
                layouts = sample_without_real(model, x_cond[:, :2], self.fixed_text_last_hidden_state, self.fixed_image_last_hidden_state, steps=self.train_dataset.max_length,
                                 temperature=1.0, sample=False, top_k=None).detach().cpu().numpy()
                sample_det_layouts = [self.train_dataset.render(layout) for layout in layouts]
                # for i, layout in enumerate(layouts):
                #     layout = self.train_dataset.render(layout)
                #     layout.save(os.path.join(self.config.samples_dir, f'sample_det_{epoch:02d}_{i:02d}.png'))
                
                # defects = []
                # for i in self.fixed_x:
                #     caption = []
                #     for j in range(0,len(i[1:(i == self.fixed_x[0][0].item()+1).nonzero(as_tuple=True)[0][0]]),5):
                #         caption.append(train_dataset.categories[i[1:(i == self.fixed_x[0][0].item()+1).nonzero(as_tuple=True)[0][0]][j].item()-224]['name'])
                #     caption = ','.join(caption)
                #     defects.append(caption)
                    
                wandb.log({
                    "input_layouts": [wandb.Image(pil, caption=f'input.png')
                                      for i, pil, in enumerate(input_layouts)],
                    "recon_layouts": [wandb.Image(pil, caption=f'recon.png')
                                      for i, pil, in enumerate(recon_layouts)],
                    "sample_random_layouts": [wandb.Image(pil, caption=f'random.png')
                                      for i, pil, in enumerate(sample_random_layouts)],
                    "sample_det_layouts": [wandb.Image(pil, caption=f'det.png')
                                      for i, pil, in enumerate(sample_det_layouts)],
                    "sample_cond_layouts": [wandb.Image(pil, caption=f'cond.png')
                                      for i, pil, in enumerate(sample_cond_layouts)],
                }, step=self.iters)
        
import seaborn as sns
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


def gen_colors(num_colors):
    """
    Generate uniformly distributed `num_colors` colors
    :param num_colors:
    :return:
    """
    palette = sns.color_palette(None, num_colors)
    rgb_triples = [[int(x[0]*255), int(x[1]*255), int(x[2]*255)] for x in palette]
    return rgb_triples


@torch.no_grad()
def sample(model, x, text, image, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.module.get_block_size() if hasattr(model, "module") else model.getcond_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
        logits, _ = model(x_cond, text, image)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x




@torch.no_grad()
def sample_without_real(model, x, text, image, steps, temperature=1.0, sample=False, top_k=None, real_x=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    
    real_x: [236,class,None, None, None, None,class, None, None, None, None]
    x:[236,class]
    """
    block_size = model.module.get_block_size() if hasattr(model, "module") else model.getcond_block_size()
    model.eval()
    for k in range(steps):
        # if (k - 1) // 5 == 0:
        #     x = torch.cat((x, real_x[:, k].reshape(-1, 1)), dim=1)
        #     continue
        if real_x is not None:
            length = x.shape[1]
            if (length-1) % 5 == 0 and length != 0 and length <= 50:
                x = torch.cat((x, real_x[:, length].reshape(-1, 1)), dim=1)
                continue 
            # 改成生成的class若不在groundtruth里才纠正？
        
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
        logits, _ = model(x_cond, text, image)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
            
        x = torch.cat((x, ix), dim=1)
    return x

@torch.no_grad()
def sample(model, x, text, image, steps, temperature=1.0, sample=False, top_k=None, real_x=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    
    real_x: [236,class,None, None, None, None,class, None, None, None, None]
    x:[236,class]
    """
    block_size = model.module.get_block_size() if hasattr(model, "module") else model.getcond_block_size()
    model.eval()
    class_status = torch.zeros(x.shape[0], x[0][0]-224 + 3).to(x.device)
    class_status[:, -3] = 1
    classes = real_x[:, 1:-1:5]
    class_total = torch.zeros(x.shape[0], x[0][0]-224 + 3).to(x.device)
    for c in range(224, x[0][0] + 2):
        class_total[:, c-224] = torch.sum(classes == c, dim=1)
            
    for k in range(steps):
        # if (k - 1) // 5 == 0:
        #     x = torch.cat((x, real_x[:, k].reshape(-1, 1)), dim=1)
        #     continue
        # if real_x is not None:
        #     length = x.shape[1]
        #     if (length-1) % 5 == 0 and length != 0 and length <= 50:
        #         x = torch.cat((x, real_x[:, length].reshape(-1, 1)), dim=1)
        #         continue 
        #     # 改成生成的class若不在groundtruth里才纠正？
        
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
        logits, _ = model(x_cond, text, image)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        
        # check
        if (x.shape[1] - 1) % 5 == 0:
            ix[ix < 224] = 224
            ix[ix >= x[0][0] + 2] = x[0][0] + 2 #[224, x[0][0] + 2]
            to_add = torch.zeros_like(class_status).to(ix.device)
            to_add = to_add.scatter(1, ix - 224, 1)
            if (class_status + to_add - class_total)[:, :-3].any(-1).sum() > 0:
                row_indice = torch.where((class_status + to_add - class_total)[:, :-1] > 0)[0] #添加多余class的样本
                # col_indice = torch.where((class_status + to_add - class_total)[:, :-1] > 0)[1]
                
                row_indice_ = torch.where((class_status - class_total)[:, :-3] < 0)[0] #仍有剩余未sample到类别class的样本
                col_indice_ = torch.where((class_status - class_total)[:, :-3] < 0)[1]
                
                row_indice__ = torch.where((class_status - class_total)[:, -2] < 0)[0] #未添加eos的样本
                # col_indice__ = torch.where((class_status - class_total)[:, -1] < 0)[1]
                
                random_select_class = []
                for row in row_indice:
                    if row in row_indice_:
                        random_select_class.append(np.random.choice(col_indice_[row_indice_== row].detach().cpu().numpy()))
                    elif row in row_indice__:
                        random_select_class.append(x[0][0]-224 + 1) #添加eos
                    else: #该样本已经sample完所有的class，此时可以添加pad
                        random_select_class.append(x[0][0]-224 + 2) #添加pad

                random_select_class = torch.tensor(random_select_class, dtype=ix.dtype).reshape(-1, 1).to(ix.device)
                
                ix[row_indice, :] = random_select_class + 224
                to_add = torch.zeros_like(class_status).to(ix.device)
                to_add = to_add.scatter(1, ix - 224, 1)
                class_status += to_add
                            
        x = torch.cat((x, ix), dim=1)
    return x

def trim_tokens(tokens, bos, eos, pad=None):
    bos_idx = np.where(tokens == bos)[0]
    tokens = tokens[bos_idx[0]+1:] if len(bos_idx) > 0 else tokens
    eos_idx = np.where(tokens == eos)[0]
    tokens = tokens[:eos_idx[0]] if len(eos_idx) > 0 else tokens
    # tokens = tokens[tokens != bos]
    # tokens = tokens[tokens != eos]
    if pad is not None:
        tokens = tokens[tokens != pad]
    return tokens



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Layout Transformer')
    parser.add_argument("--exp", default="layout", help="experiment name")
    parser.add_argument("--log_dir", default="./logs", help="/path/to/logs/dir")

    # MNIST options
    parser.add_argument("--data_dir", default=None, help="/path/to/mnist/data")
    parser.add_argument("--threshold", type=int, default=16, help="threshold for grayscale values")

    # COCO/PubLayNet options
    parser.add_argument("--train_json", default="/Data4/student_zhihan_data/Diffusion/annotations/panoptic_train2017.json", help="/path/to/train/json")
    parser.add_argument("--val_json", default="/Data4/student_zhihan_data/Diffusion/annotations/panoptic_val2017.json", help="/path/to/val/json")

    # Layout options
    parser.add_argument("--max_length", type=int, default=128, help="batch size")
    parser.add_argument('--precision', default=8, type=int)
    parser.add_argument('--element_order', default='raster')
    parser.add_argument('--attribute_order', default='cxywh')

    # Architecture/training options
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=4.5e-06, help="learning rate")
    parser.add_argument('--n_layer', default=6, type=int)
    parser.add_argument('--n_embd', default=512, type=int)
    parser.add_argument('--n_head', default=8, type=int)
    # parser.add_argument('--evaluate', action='store_true', help="evaluate only")
    parser.add_argument('--lr_decay', action='store_true', help="use learning rate decay")
    parser.add_argument('--warmup_iters', type=int, default=0, help="linear lr warmup iters")
    parser.add_argument('--final_iters', type=int, default=0, help="cosine lr final iters")
    parser.add_argument('--sample_every', type=int, default=1, help="sample every epoch")
    parser.add_argument('--project_name', default="PCB_layout", help="wandb project name")
    parser.add_argument('--image', action='store_true', help="use image as input")
    parser.add_argument('--text', action='store_true', help="use text as input")
    parser.add_argument('--sample_with_text', action='store_true', help="use image and text as input")
    parser.add_argument('--pretrained_ckpt', default=None, help="/path/to/pretrained/ckpt")
    
    args = parser.parse_args()

    log_dir = os.path.join(args.log_dir, args.exp)
    samples_dir = os.path.join(log_dir, "samples")
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # train_dataset = JSONLayout('/Data4/student_zhihan_data/source_code/Mask-free-Defect-Generation/Layout-Generation/dataset/GC10/train/_annotations.coco.json', max_length=50)
    train_dataset = JSONLayout('/Data4/student_zhihan_data/source_code/Mask-free-Defect-Generation/Layout-Generation/dataset/PCB/train/_annotations.coco.json', max_length=50)
    # valid_dataset = JSONLayout('/Data4/student_zhihan_data/source_code/Mask-free-Defect-Generation/Layout-Generation/dataset/GC10/valid/_annotations.coco.json', max_length=50)
    valid_dataset = JSONLayout('/Data4/student_zhihan_data/source_code/Mask-free-Defect-Generation/Layout-Generation/dataset/PCB/valid/_annotations.coco.json', max_length=50)
    # train_dataset = COCOLayout(args.train_json, max_length=50)
    # valid_dataset = COCOLayout(args.val_json, max_length=50)

    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_length,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, image_available=args.image, text_available=args.text, pretrained_ckpt=args.pretrained_ckpt)  # a GPT-1
    model = GPT(mconf)
    
    tconf = TrainerConfig(max_epochs=args.epochs,
                          batch_size=args.batch_size,
                          lr_decay=args.lr_decay,
                          learning_rate=args.lr * args.batch_size,
                          warmup_iters=args.warmup_iters,
                          final_iters=args.final_iters,
                          ckpt_dir=ckpt_dir,
                          samples_dir=samples_dir,
                          sample_every=args.sample_every, pretrained_ckpt=args.pretrained_ckpt)
    trainer = Trainer(model, train_dataset, valid_dataset, tconf, args)
    trainer.train()
    # trainer.pretrain()
