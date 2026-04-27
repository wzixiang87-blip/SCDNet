import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.SparseClusterDecomposition import ConstrainedSparseClusterDecomposition
from layers.RevIN import RevIN



class StableFusion(nn.Module):
    def __init__(self, pred_len):
        super().__init__()
        self.linear = nn.Linear(pred_len * 2, pred_len)
        self.sigmoid = nn.Sigmoid()


        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, common, residual):


        B, L, N = common.shape


        common_t = common.transpose(1, 2)
        residual_t = residual.transpose(1, 2)

        # 拼接两路
        fusion_input = torch.cat([common_t, residual_t], dim=-1)


        fusion_input = torch.clamp(fusion_input, -10.0, 10.0)

        gate = self.sigmoid(self.linear(fusion_input))


        gate = gate.transpose(1, 2)


        fused = common + gate * residual

        return fused



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.d_model = configs.d_model
        self.cycle_len = configs.cycle
        self.enc_in = configs.enc_in

        self.use_gated_attention = getattr(configs, 'use_gated_attention', False)


        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)


        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )


        n_clusters = getattr(configs, 'n_clusters', 32)
        top_k = getattr(configs, 'top_k', 4)


        self.scd_module = ConstrainedSparseClusterDecomposition(
            d_model=configs.d_model,
            n_clusters=n_clusters,
            top_k=top_k,
            seq_len=configs.seq_len,
            pred_len=configs.pred_len
        )


        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ),
                        configs.d_model,
                        configs.n_heads,
                        gated_attention=self.use_gated_attention
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )


        self.projector_common = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model * 2),
            nn.GELU(),
            nn.Dropout(configs.output_proj_dropout),
            nn.Linear(configs.d_model * 2, configs.pred_len)
        )

        self.projector_residual = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.GELU(),
            nn.Dropout(configs.output_proj_dropout),
            nn.Linear(configs.d_model, configs.pred_len)
        )


        self.fusion = StableFusion(configs.pred_len)


        self.channel_embedding = nn.Parameter(
            torch.zeros(configs.enc_in, configs.d_model)
        )

        self.phase_embedding = nn.Embedding(configs.cycle, configs.d_model)

        self.joint_embedding = nn.Embedding(
            configs.cycle,
            configs.enc_in * configs.d_model
        )

        nn.init.xavier_normal_(self.phase_embedding.weight)
        nn.init.xavier_normal_(self.joint_embedding.weight)
        nn.init.xavier_normal_(self.channel_embedding)


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, phase):


        if self.use_norm:
            x_enc = self.revin_layer(x_enc, 'norm')

        B, L, N = x_enc.shape


        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        channel_emb = self.channel_embedding.expand(B, N, -1)
        phase_emb = self.phase_embedding(
            phase.view(-1, 1).expand(B, N)
        )

        joint_emb = self.joint_embedding(phase).reshape(
            B, self.enc_in, self.d_model
        )

        enc_out = enc_out[:, :N, :] + channel_emb + phase_emb + joint_emb


        x_common, x_residual, aux_loss = self.scd_module(enc_out)


        enc_common_out, attns = self.encoder(x_common, attn_mask=None)

        dec_out_common = self.projector_common(
            enc_common_out
        ).permute(0, 2, 1)[:, :, :N]

        #
        dec_out_residual = self.projector_residual(
            x_residual
        ).permute(0, 2, 1)[:, :, :N]


        dec_out = self.fusion(dec_out_common, dec_out_residual)


        if self.use_norm:
            dec_out = self.revin_layer(dec_out, 'denorm')

        return dec_out, attns, aux_loss


    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        cycle_index,
        mask=None
    ):
        dec_out, attns, aux_loss = self.forecast(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec,
            cycle_index
        )

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns, aux_loss
        else:
            return dec_out[:, -self.pred_len:, :], aux_loss