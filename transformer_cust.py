import math
import logging
from typing import Sequence, Union

import torch
import torch.nn as nn

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.nets import ViT

from einops import repeat, rearrange
from timm.models.layers import DropPath, trunc_normal_

class transformer_cust(nn.Module):
    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        dropout_rate: float = 0.0,
        decoder_dim: int = 768,
        decoder_depth: int = 1,
        decoder_heads: int = 8,
        masking_ratio: float = 0.75,
        **kwargs,
    ):
        
        super().__init__()

        self.encoder = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
        )

        self.image_size = img_size
        self.patch_size = patch_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        # patch embedding block

        patch_embedding = self.encoder.patch_embedding
        self.to_patch = patch_embedding.patch_embeddings
        #self.to_patch, self.patch_to_emb = patch_embedding.patch_embeddings #perceptron patch
        n_patches = patch_embedding.n_patches
        patch_dim = patch_embedding.patch_dim
        self.encoder_pos = nn.Parameter(torch.zeros(n_patches, hidden_size))


        #connect encoder and decoder if mismatch dimension
        self.enc_to_dec = (
            nn.Linear(hidden_size, decoder_dim)
            if hidden_size != decoder_dim
            else nn.Identity()
        )

        # build up decoder transformer blocks
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    decoder_dim, decoder_dim * 4, decoder_heads, dropout_rate
                )
                for i in range(decoder_depth)
            ]
        )
        self.encoder_norm = nn.LayerNorm(hidden_size)
        self.decoder_norm = nn.LayerNorm(decoder_dim)

        self.masking_ratio = masking_ratio
        assert (
            masking_ratio > 0 and masking_ratio < 1
        ), "masking ratio must be kept between 0 and 1"
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder_pos_emb = nn.Embedding(n_patches, decoder_dim)

        # embeddings to pixels
        self.to_pixels = nn.Linear(decoder_dim, patch_dim)

        self.init_weights()

    def init_weights(self):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.pos_embed, std=0.02)
            nn.init.normal_(self.decoder_pos_embed, std=0.02)
            nn.init.normal_(self.mask_token, std=0.02)


    def forward(self, x):
        """
        Args:
            x: input tensor must have isotropic spatial dimensions,
                such as ``[batch_size, channels, sp_size, sp_size[, sp_size]]``.
        """
        device = x.device
       
        # get patches
        patches = self.to_patch(x)
        
        patches = rearrange(patches, 'b D H W C  -> b (H W C) D ')
        
        x=rearrange(x, 'b D (h p1) (w p2) (c p3)  -> b (h w c) (p1 p2 p3 D) ',p1=self.patch_size[0], p2=self.patch_size[1], p3=self.patch_size[2])

        batch, n_patches, *_ = patches.shape #perceptron
        token_xdim = self.image_size[0] // self.patch_size[0]
        token_ydim = self.image_size[1] // self.patch_size[1]
        token_zdim = self.image_size[2] // self.patch_size[2]
        # patch and cls token to encoder tokens and add positions
        #patches = self.patch_to_emb(patches) #perceptron
        patches = patches + self.encoder_pos
        CLS_token = repeat(self.cls_token, '() n e -> b n e', b=batch)
        tokens = torch.cat([CLS_token, patches], dim=1)
        
        # patches = patches + self.encoder.patch_embedding.position_embeddings


        # calculate of patches needed to be masked, and get random indices
        num_masked = int(self.masking_ratio * n_patches)
        rand_indices = torch.rand(batch, n_patches, device=device).argsort(dim=-1)
        ids_restore = torch.argsort(rand_indices, dim=1)
        masked_indices, unmasked_indices = (
            rand_indices[:, :num_masked],
            rand_indices[:, num_masked:],
        )  #B, masked token / B, unmaksed token 
        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device=device)[:, None]
        x_ = tokens[:,1:,:]
        
        x_ = x_[batch_range, unmasked_indices]  #B,unmasked token, dim
        tokens = torch.cat([tokens[:,0:1,:],x_],dim=1)

        #get the patches to be masked for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices]
        
        for blk in self.encoder.blocks:
            tokens = blk(tokens)
        tokens = self.encoder_norm(tokens)
        encoded_tokens = tokens

        decoder_tokens = self.enc_to_dec(encoded_tokens)
        decoder_tokens_ = decoder_tokens[:,1:,:]
        decoder_tokens_ += self.decoder_pos_emb(unmasked_indices)

        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens_ = torch.cat((mask_tokens, decoder_tokens_), dim=1)  #mask + unmask
        decoder_tokens = torch.cat((decoder_tokens[:,0:1,:],decoder_tokens_), dim=1)

        for blk in self.decoder_blocks:
            decoder_tokens = blk(decoder_tokens)
        decoded_tokens = self.decoder_norm(decoder_tokens)

        # splice out the mask tokens and project to pixel values
        decoded_tokens = decoded_tokens[:,1:,:]

        mask_tokens = decoded_tokens[:, :num_masked, :]
        decoded_tokens = self.to_pixels(decoded_tokens)

        pred_pixel_values_mask = decoded_tokens[:,:num_masked,:]
        pred_pixel_values_unmask = decoded_tokens[:,num_masked:,:]

        recon = torch.gather(decoded_tokens, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, decoded_tokens.shape[2]))
        #rearranged_img = rearrange(decoded_tokens, 'b (h w d) (P1 P2 P3 C) -> b C (h P1) (w P2) (d P3) ', h=dim,w=dim,d=dim,P1=patch_size,P2=patch_size,P3=patch_size,C=1)    
        rearranged_img = rearrange(recon, 'b (h w d) (P1 P2 P3 C) -> b C (h P1) (w P2) (d P3) ', h=token_xdim,w=token_ydim,d=token_zdim,P1=self.patch_size[0],P2=self.patch_size[1],P3=self.patch_size[2],C=1)
        

        #return decoded_tokens, x, rearranged_img
        return pred_pixel_values_mask, pred_pixel_values_unmask, x, batch_range, masked_indices, unmasked_indices, rearranged_img
    

class transformer_cust_classification_task(nn.Module):
    def __init__(
        self,
        in_channels: int,
        MRI_img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        dropout_rate: float = 0.0,
        class_num: int = 2,
        **kwargs,
    ):
        
        super().__init__()

        self.encoder = ViT(
            in_channels=in_channels,
            img_size=MRI_img_size, 
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        # patch embedding block
        self.patch_size = patch_size

        patch_embedding = self.encoder.patch_embedding
        self.to_patch = patch_embedding.patch_embeddings
        #self.to_patch, self.patch_to_emb = patch_embedding.patch_embeddings #perceptron patch
        n_patches = patch_embedding.n_patches
        patch_dim = patch_embedding.patch_dim
        self.encoder_pos = nn.Parameter(torch.zeros(n_patches, hidden_size))

        self.encoder_norm = nn.LayerNorm(hidden_size)

        # classification task
        self.class_head = nn.Linear(hidden_size,class_num)

        self.init_weights()

    def init_weights(self):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)


    def forward(self, x):
        """
        Args:
            x: input tensor must have isotropic spatial dimensions,
                such as ``[batch_size, channels, sp_size, sp_size[, sp_size]]``.
        """
        device = x.device

        # get patches

        patches = self.to_patch(x)
        
        patches = rearrange(patches, 'b D H W C  -> b (H W C) D ')
        
        x=rearrange(x, 'b D (h p1) (w p2) (c p3)  -> b (h w c) (p1 p2 p3 D) ',p1=self.patch_size[0], p2=self.patch_size[1], p3=self.patch_size[2])

        batch, n_patches, *_ = patches.shape #perceptron

        # patch and cls token to encoder tokens and add positions
        patches = patches + self.encoder_pos
        CLS_token = repeat(self.cls_token, '() n e -> b n e', b=batch)
        tokens = torch.cat([CLS_token, patches], dim=1)
        

        # patch to encoder tokens and add positions

        for blk in self.encoder.blocks:
            tokens = blk(tokens)
        encoded_tokens = tokens # B,L,D
        tokens = self.encoder_norm(tokens)

        # classification task
        pred_class = self.class_head(encoded_tokens[:,0,:])

        return pred_class
    



class transformer_cust_prediction_task(nn.Module):
    def __init__(
        self,
        in_channels: int,
        MRI_img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        dropout_rate: float = 0.0,
        class_num: int = 2,
        **kwargs,
    ):
        
        super().__init__()

        self.encoder = ViT(
            in_channels=in_channels,
            img_size=MRI_img_size, 
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        # patch embedding block
        self.patch_size = patch_size
        
        patch_embedding = self.encoder.patch_embedding
        self.to_patch = patch_embedding.patch_embeddings
        #self.to_patch, self.patch_to_emb = patch_embedding.patch_embeddings #perceptron patch
        n_patches = patch_embedding.n_patches
        patch_dim = patch_embedding.patch_dim
        self.encoder_pos = nn.Parameter(torch.zeros(n_patches, hidden_size))

        self.encoder_norm = nn.LayerNorm(hidden_size)

        # classification task
        self.prediction_head = nn.Linear(hidden_size,1)

        self.init_weights()

    def init_weights(self):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)


    def forward(self, x):
        """
        Args:
            x: input tensor must have isotropic spatial dimensions,
                such as ``[batch_size, channels, sp_size, sp_size[, sp_size]]``.
        """
        device = x.device

        # get patches

        patches = self.to_patch(x)
        
        patches = rearrange(patches, 'b D H W C  -> b (H W C) D ')
        
        x=rearrange(x, 'b D (h p1) (w p2) (c p3)  -> b (h w c) (p1 p2 p3 D) ',p1=self.patch_size[0], p2=self.patch_size[1], p3=self.patch_size[2])

        batch, n_patches, *_ = patches.shape #perceptron

        # patch and cls token to encoder tokens and add positions
        patches = patches + self.encoder_pos
        CLS_token = repeat(self.cls_token, '() n e -> b n e', b=batch)
        tokens = torch.cat([CLS_token, patches], dim=1)
        

        # patch to encoder tokens and add positions

        for blk in self.encoder.blocks:
            tokens = blk(tokens)
        encoded_tokens = tokens # B,L,D
        tokens = self.encoder_norm(tokens)

        # classification task
        prediction = self.prediction_head(encoded_tokens[:,0,:])

        return prediction