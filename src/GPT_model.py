import torch
import torch.nn as nn
from attention import MultiHeadAttention
import os
import urllib.request
from tqdm import tqdm
import tensorflow as tf
import json
import numpy as np

# GPT Implementation and all other classes in this file are straight from Rachka's book
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
 
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
 
    def forward(self, x):
        return self.layers(x)
 
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
 
    def forward(self, x):

        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
 
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
       
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
 
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
    def load_pretrained_weights(self, model_size, models_dir):
        model_dir = os.path.join(models_dir, model_size)

        self.__download_pretrained_model__(model_size, model_dir)

        # Load settings and params
        tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
        settings = json.load(open(os.path.join(model_dir, "hparams.json")))
        params = self.__load_params_from_tf_ckpt__(tf_ckpt_path, settings)

        self.__load_weights_into_gpt__(params)
        return params
    
    def __assign__(self, left, right):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(torch.tensor(right))
    
    def __download_file__(self, url, destination):
        # Send a GET request to download the file
        with urllib.request.urlopen(url) as response:
            # Get the total file size from headers, defaulting to 0 if not present
            file_size = int(response.headers.get("Content-Length", 0))

            # Check if file exists and has the same size
            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"File already exists and is up-to-date: {destination}")
                    return

            # Define the block size for reading the file
            block_size = 1024  # 1 Kilobyte

            # Initialize the progress bar with total file size
            progress_bar_description = os.path.basename(url)  # Extract filename from URL
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
                # Open the destination file in binary write mode
                with open(destination, "wb") as file:
                    # Read the file in chunks and write to destination
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress_bar.update(len(chunk))  # Update progress bar
    
    def __download_pretrained_model__(self, model_size, model_dir):
        print('Currently, this module will only load from the gpt-2 public model.  In the future, the plan is to allow you to load any pretrained weights')
        # Validate model size
        allowed_sizes = ("124M", "355M", "774M", "1558M")
        if model_size not in allowed_sizes:
            raise ValueError(f"Model size not in {allowed_sizes}")

        # Define paths
        base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
        filenames = [
            "checkpoint", "encoder.json", "hparams.json",
            "model.ckpt.data-00000-of-00001", "model.ckpt.index",
            "model.ckpt.meta", "vocab.bpe"
        ]

        # Download files
        os.makedirs(model_dir, exist_ok=True)
        for filename in filenames:
            file_url = os.path.join(base_url, model_size, filename)
            file_path = os.path.join(model_dir, filename)
            self.__download_file__(file_url, file_path)
    
    def __load_params_from_tf_ckpt__(self, ckpt_path, settings):
        # Initialize parameters dictionary with empty blocks for each layer
        params = {"blocks": [{} for _ in range(settings["n_layer"])]}

        # Iterate over each variable in the checkpoint
        for name, _ in tf.train.list_variables(ckpt_path):
            # Load the variable and remove singleton dimensions
            variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

            # Process the variable name to extract relevant parts
            variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

            # Identify the target dictionary for the variable
            target_dict = params
            if variable_name_parts[0].startswith("h"):
                layer_number = int(variable_name_parts[0][1:])
                target_dict = params["blocks"][layer_number]

            # Recursively access or create nested dictionaries
            for key in variable_name_parts[1:-1]:
                target_dict = target_dict.setdefault(key, {})

            # Assign the variable array to the last key
            last_key = variable_name_parts[-1]
            target_dict[last_key] = variable_array

        return params
    
    def __load_weights_into_gpt__(self, params):
        self.pos_emb.weight = self.__assign__(self.pos_emb.weight, params['wpe'])
        self.tok_emb.weight = self.__assign__(self.tok_emb.weight, params['wte'])

        for b in range(len(params["blocks"])):
            q_w, k_w, v_w = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
            self.trf_blocks[b].att.W_query.weight = self.__assign__(
                self.trf_blocks[b].att.W_query.weight, q_w.T)
            self.trf_blocks[b].att.W_key.weight = self.__assign__(
                self.trf_blocks[b].att.W_key.weight, k_w.T)
            self.trf_blocks[b].att.W_value.weight = self.__assign__(
                self.trf_blocks[b].att.W_value.weight, v_w.T)

            q_b, k_b, v_b = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
            self.trf_blocks[b].att.W_query.bias = self.__assign__(
                self.trf_blocks[b].att.W_query.bias, q_b)
            self.trf_blocks[b].att.W_key.bias = self.__assign__(
                self.trf_blocks[b].att.W_key.bias, k_b)
            self.trf_blocks[b].att.W_value.bias = self.__assign__(
                self.trf_blocks[b].att.W_value.bias, v_b)

            self.trf_blocks[b].att.out_proj.weight = self.__assign__(
                self.trf_blocks[b].att.out_proj.weight,
                params["blocks"][b]["attn"]["c_proj"]["w"].T)
            self.trf_blocks[b].att.out_proj.bias = self.__assign__(
                self.trf_blocks[b].att.out_proj.bias,
                params["blocks"][b]["attn"]["c_proj"]["b"])

            self.trf_blocks[b].ff.layers[0].weight = self.__assign__(
                self.trf_blocks[b].ff.layers[0].weight,
                params["blocks"][b]["mlp"]["c_fc"]["w"].T)
            self.trf_blocks[b].ff.layers[0].bias = self.__assign__(
                self.trf_blocks[b].ff.layers[0].bias,
                params["blocks"][b]["mlp"]["c_fc"]["b"])
            self.trf_blocks[b].ff.layers[2].weight = self.__assign__(
                self.trf_blocks[b].ff.layers[2].weight,
                params["blocks"][b]["mlp"]["c_proj"]["w"].T)
            self.trf_blocks[b].ff.layers[2].bias = self.__assign__(
                self.trf_blocks[b].ff.layers[2].bias,
                params["blocks"][b]["mlp"]["c_proj"]["b"])

            self.trf_blocks[b].norm1.scale = self.__assign__(
                self.trf_blocks[b].norm1.scale,
                params["blocks"][b]["ln_1"]["g"])
            self.trf_blocks[b].norm1.shift = self.__assign__(
                self.trf_blocks[b].norm1.shift,
                params["blocks"][b]["ln_1"]["b"])
            self.trf_blocks[b].norm2.scale = self.__assign__(
                self.trf_blocks[b].norm2.scale,
                params["blocks"][b]["ln_2"]["g"])
            self.trf_blocks[b].norm2.shift = self.__assign__(
                self.trf_blocks[b].norm2.shift,
                params["blocks"][b]["ln_2"]["b"])

        self.final_norm.scale = self.__assign__(self.final_norm.scale, params["g"])
        self.final_norm.shift = self.__assign__(self.final_norm.shift, params["b"])
        self.out_head.weight = self.__assign__(self.out_head.weight, params["wte"])