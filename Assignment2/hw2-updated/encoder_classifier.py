#!/usr/bin/env python
# -*- coding: utf-8 -*-
#   
# Copyright (C) 2024
# 
# @author: prachi@andrew.cmu.edu, fhammed@andrew.cmu.edu

"""
11-411/611 NLP Assignment 2
Transformer Encoder Classifier Implementation

Complete the PositionalEncoding and TransformerEncoderClassifier classes.

Task: Record Classification
- Classify song lyrics into one of three records (s1, s2, s3)
- s1: "The Tortured Poets Department"
- s2: "So Long, London"
- s3: "Down Bad"

You are provided with:
- Pre-trained GloVe embeddings (50-dimensional) as an embedding_matrix
- Data loading and preprocessing utilities

You need to implement:
- PositionalEncoding: Add positional information to embeddings
- TransformerEncoderClassifier: The main encoder model for classification
"""

#######################################
# Import Statements
#######################################
import torch
import torch.nn as nn
import math


#######################################
# TODO: PositionalEncoding()
#######################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        """
        Positional Encoding module that adds positional information to embeddings.

        Args
        ____
        d_model: int
            Dimension of the model (embedding dimension, should be 50 for GloVe)
        max_len: int
            Maximum sequence length
        dropout: float
            Dropout rate

        HINT: Use sine and cosine functions of different frequencies.
        For position pos and dimension i:
            PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        Store the positional encodings as a buffer (not a parameter) using:
            self.register_buffer('pe', pe_tensor)
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        self.register_buffer('pe', pe)


    def forward(self, x):
        """
        Add positional encoding to input embeddings.

        Args
        ____
        x: torch.Tensor
            Input tensor of shape (batch_size, sequence_length, d_model)

        Returns
        -------
        torch.Tensor
            Output tensor with positional encoding added, same shape as input
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


#######################################
# TODO: TransformerEncoderClassifier()
#######################################
class TransformerEncoderClassifier(nn.Module):
    def __init__(self, embedding_matrix, num_classes, nhead=2, num_layers=2,
                 dim_feedforward=128, dropout=0.1, max_seq_len=64):
        """
        Transformer Encoder for record classification.

        Args
        ____
        embedding_matrix: torch.Tensor
            Pre-trained embedding matrix of shape (vocab_size, d_model)
            Use this directly - do NOT create a new nn.Embedding from scratch
        num_classes: int
            Number of output classes (3 for our records: s1, s2, s3)
        nhead: int
            Number of attention heads (d_model must be divisible by nhead)
        num_layers: int
            Number of transformer encoder layers
        dim_feedforward: int
            Dimension of the feedforward network in each encoder layer
        dropout: float
            Dropout rate
        max_seq_len: int
            Maximum sequence length

        Attributes to create:
            self.embedding: nn.Embedding
                Initialize from embedding_matrix using nn.Embedding.from_pretrained()
                Set freeze=False to allow fine-tuning
            self.d_model: int
                Embedding dimension (get from embedding_matrix.shape[1])
            self.positional_encoding: PositionalEncoding
                Your positional encoding module
            self.transformer_encoder: nn.TransformerEncoder
                Stack of transformer encoder layers
                Use nn.TransformerEncoderLayer and nn.TransformerEncoder
            self.fc: nn.Linear
                Classification head (d_model -> num_classes)
            self.dropout: nn.Dropout
                Dropout layer

        Note: You may use nn.TransformerEncoderLayer and nn.TransformerEncoder from PyTorch
        """
        super(TransformerEncoderClassifier, self).__init__()

        self.device = torch.device("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available()
                                 else "cpu")
        print(f"Using device: {self.device}")

        # TODO: Your code here
        # 1. Store d_model from embedding_matrix shape
        self.d_model = embedding_matrix.shape[1]
        # 2. Create embedding layer from pre-trained embedding_matrix
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        # 3. Initialize positional encoding
        self.positional_encoding = PositionalEncoding(d_model=self.d_model, max_len=max_seq_len, dropout=dropout)
        # 4. Initialize transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 5. Initialize classification head (fc layer)
        self.fc = nn.Linear(self.d_model, num_classes)
        # 6. Initialize dropout
        self.dropout = nn.Dropout(dropout)

        # End of your code
        self.to(self.device)



    def forward(self, x, attention_mask=None):
        """
        Forward pass of the transformer encoder classifier.

        Args
        ____
        x: torch.Tensor
            Input tensor of shape (batch_size, sequence_length) containing token indices
        attention_mask: torch.Tensor
            Attention mask of shape (batch_size, sequence_length)
            1 for real tokens, 0 for padding tokens

        Returns
        -------
        logits: torch.Tensor
            Output logits of shape (batch_size, num_classes)

        HINTS:
        1. Get embeddings from self.embedding(x)
        2. Scale embeddings by sqrt(d_model) (helps with training stability)
        3. Add positional encoding
        4. Transpose for transformer: (batch, seq, dim) -> (seq, batch, dim)
        5. Create key_padding_mask for transformer: True where padding, False where real tokens
           (This is the OPPOSITE of attention_mask!)
        6. Pass through transformer encoder
        7. Transpose back: (seq, batch, dim) -> (batch, seq, dim)
        8. Pool the sequence (mean pooling over non-padded tokens recommended)
        9. Pass through classification head

        Note: Don't forget to move tensors to the correct device!
        """
        x = x.to(self.device)
        embedded = self.embedding(x)
        embedded = embedded * math.sqrt(self.d_model)
        embedded = self.positional_encoding(embedded)
        embedded = embedded.transpose(0, 1)  # (batch, seq, dim) -> (seq, batch, dim)

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            key_padding_mask = (attention_mask == 0).bool()  # Boolean mask
        else:
            key_padding_mask = None

        encoded = self.transformer_encoder(embedded, src_key_padding_mask=key_padding_mask)
        encoded = encoded.transpose(0, 1)  # (seq, batch, dim) -> (batch, seq, dim)
        # Mean pooling over non-padded tokens
        if attention_mask is not None:
            # Expand attention mask to match the embedding dimension: (batch_size, seq_len, d_model)
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(encoded).float()
            
            # Sum the embeddings for real tokens
            sum_embeddings = torch.sum(encoded * mask_expanded, dim=1)
            
            # Count the number of real tokens (clamp to 1e-9 to avoid division by zero)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            
            # Mean pooling
            pooled = sum_embeddings / sum_mask
        else:
            # Fallback if no mask is provided
            pooled = encoded.mean(dim=1)

        pooled = self.dropout(pooled)
        logits = self.fc(pooled)

        return logits


    def predict(self, x, attention_mask=None):
        """
        Predict class labels for input sequences.

        Args
        ____
        x: torch.Tensor
            Input tensor of shape (batch_size, sequence_length)
        attention_mask: torch.Tensor
            Attention mask of shape (batch_size, sequence_length)

        Returns
        -------
        predictions: torch.Tensor
            Predicted class indices of shape (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, attention_mask)
            predictions = torch.argmax(logits, dim=1)
        return predictions

