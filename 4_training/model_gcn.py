import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_pro, hidden_dim):
        super(TransformerEncoderBlock, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_pro = dropout_pro
        self.hidden_dim = hidden_dim
        # Self-Attention
        self.query_projection = nn.Linear(self.embed_dim, self.embed_dim,
                                          bias=False)
        self.key_projection = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.value_projection = nn.Linear(self.embed_dim, self.embed_dim,
                                          bias=False)
        self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads,
                                               dropout=self.dropout_pro,
                                               batch_first=True)

        # Feed Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.embed_dim)
        )

        # Layer Normalization, added second one 8/18
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        # Self Attention
        residual1 = x
        transpose = x
        query = self.query_projection(transpose)
        key = self.key_projection(transpose)
        value = self.value_projection(transpose)
        attn_output, _ = self.attention(query, key, value)
        sum1 = residual1 + attn_output  # Add & Norm
        layernorm1 = self.layer_norm1(sum1)
        # Feed Forward
        residual2 = layernorm1
        feed_forward = self.feed_forward(residual2)
        sum2 = feed_forward + residual2  # Add & Norm
        output = self.layer_norm2(sum2)  # Layer Norm
        return output


class RuptureNet2D(nn.Module):
    def __init__(self, config):
        super().__init__()

        # GCN
        gcn_in = config.get('gcn_in_features',4)
        gcn_hidden = config.get('gcn_hidden_dim',128)
        gcn_out = config.get('gcn_out_dim',256)
        gcn_num_layers = 4

        self.gcn_layers = nn.ModuleList()

        for i in range(gcn_num_layers):
            in_dim = gcn_in if i==0 else gcn_hidden
            out_dim = gcn_out if i==gcn_num_layers - 1 else gcn_hidden
            self.gcn_layers.append(GCNConv(in_dim, out_dim))


        # input convolution, 4 input features
        self.conv_input = nn.Conv1d(gcn_out, config['conv1_dim'], kernel_size=1, stride=1, padding=0)

        # first convolution block
        self.conv1 = nn.ModuleList()
        cur_dim = config['conv1_dim']
        for i in range(config['conv1_layers']):
            # square dimension in each step for concatination
            self.conv1.append(nn.Conv1d(cur_dim, cur_dim, kernel_size=1, stride=1, padding=0))

        # Added 8/18
        self.embed_dim = cur_dim * (1 + config['conv1_layers'])

        # Added 3/5 for downsampling, 1890 -> 944
        self.down_conv1 = nn.Conv1d(
            self.embed_dim,
            self.embed_dim,
            kernel_size=3,
            stride=2,
            padding=0
        )

        # Second downsampling, not used
        self.down_conv2 = nn.Conv1d(
            self.embed_dim,
            self.embed_dim,
            kernel_size=3,
            stride=2,
            padding=0
        )

        # Transformer blocks
        self.attention_layers = nn.ModuleList()
        for i in range(config['num_transformer_blocks']):
            self.attention_layers.append(
                TransformerEncoderBlock(
                    embed_dim=self.embed_dim,   #cur_dim, edited 8/18
                    num_heads=config['num_transformer_heads'],
                    dropout_pro=config['dropout_pro'],
                    hidden_dim=config['transformer_hidden_dim']
                )
            )

        # second convolution layers, removed 5/4 to separate decoder
            # self.conv2 = nn.ModuleList()
            # # Changed cur_dim to self.embed_dim 8/18
            # self.conv2.append(nn.Conv1d(self.embed_dim, config['conv2_dim'], kernel_size=1, stride=1, padding=0))

            # self.conv_output = nn.Conv1d(config['conv2_dim'], 3, kernel_size=1, stride=1, padding=0)

        # Added 8/28 to fix dimensions
        # self.proj_to_transformer = nn.Linear(self.embed_dim, config['transformer_hidden_dim'])

        # Separate decoder branches, added 5/4/26
        conv2_dim = config['conv2_dim']

        self.conv2_time = nn.Conv1d(self.embed_dim, conv2_dim, kernel_size=1)
        self.conv2_height = nn.Conv1d(self.embed_dim, conv2_dim, kernel_size=1)
 
        self.head_time = nn.Conv1d(conv2_dim, 1, kernel_size=1)
        self.head_height = nn.Conv1d(conv2_dim, 1, kernel_size=1)
        self.head_inundate = nn.Conv1d(conv2_dim, 1, kernel_size=1)

        self.relu = nn.ReLU()

        # Learned linear interpolation, added 8/13/26
        self.coastal_projection = nn.LazyLinear(520)

        # for scheduler, removed 8/18
        #self.embed_dim = cur_dim

    def forward(self, x, edge_index):
        """
        Args:
            data: torch_geometric.data.Batch
                data.x          — (total_nodes, 4)  node features
                data.edge_index — (2, total_edges)   COO edge list
                data.batch      — (total_nodes,)     sample index per node
        Returns:
            (N, 670, 3)
        """
        batch_size = x.size(0)
        num_nodes = x.size(1)
   
        x_flat = x.view(-1, x.size(-1))

        # --- GCN encoder ---
        # (total_nodes, 4) -> (total_nodes, gcn_out)
        for i, gcn in enumerate(self.gcn_layers):
            x_flat = gcn(x_flat, edge_index)
            if i < len(self.gcn_layers) - 1:
                x_flat = self.relu(x_flat)
        # no activation on final GCN layer — conv_input acts as the nonlinearity

        # Reshape flat PyG batch -> (N, num_nodes, gcn_out)
        # Safe because all samples share the same fixed mesh.
        # num_nodes = x.size(0) // batch_size
        x = x_flat.view(batch_size, num_nodes, -1)

        # CNN Encoder
        # (N, num_nodes, gcn_out) -> (N, gcn_out, num_nodes)
        x = torch.transpose(x, 1, 2)

        # (N, gcn_out, num_nodes) -> (N, conv1_dim, num_nodes)
        x = self.relu(self.conv_input(x))

        conv_outputs = [x]
        for layer in self.conv1:
            new = self.relu(layer(x))
            conv_outputs.append(new)
        x = torch.cat(conv_outputs, dim=1)

        # Down conv
        x = self.down_conv1(x)
        #x = self.down_conv2(x)

        # (N, self.embed_dim, ~950) -> (N, ~950, self.embed_dim)
        x = torch.transpose(x, 1, 2)

        # Transformer Processor
        for layer in self.attention_layers:
            x = layer(x)

        # CNN Decoder
        # (N, ~950, self.embed_dim) -> (N, self.embed_dim, ~950)
        x = torch.transpose(x, 1, 2)

        # Removed 5/4 for separate branches
            # for layer in self.conv2:
            #     x = self.relu(layer(x))

            # x = self.conv_output(x)

        # CNN Decoder
        time_features = self.relu(self.conv2_time(x))
        time_out = self.head_time(time_features)

        height_features = self.relu(self.conv2_height(x))
        # Added self.relu below, 8/13/2026
        height_out = self.head_height(height_features)

        inundate_out = self.head_inundate(height_features)

        # Interpolate to 520 features, apply relu
        time_out = self.coastal_projection(time_out)
        height_out = self.relu(self.coastal_projection(height_out))
        inundate_out = self.coastal_projection(inundate_out)

        x = torch.cat([time_out, height_out, inundate_out],dim=1)


        # (N, 3, 670) -> (N, 670, 3)
        x = torch.transpose(x, 1, 2)
        return x    
