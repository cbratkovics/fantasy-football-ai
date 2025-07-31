"""
Advanced Neural Network Architectures
LSTM, Transformer, and Attention-based models for fantasy football prediction
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MultiHeadSelfAttention(layers.Layer):
    """Multi-head self-attention layer for Transformer models"""
    
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        assert embed_dim % num_heads == 0
        
        self.head_dim = embed_dim // num_heads
        
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        
        self.combine_heads = layers.Dense(embed_dim)
        self.dropout_layer = layers.Dropout(dropout)
        
    def attention(self, query, key, value, mask=None):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        
        if mask is not None:
            scaled_score += (mask * -1e9)
        
        weights = tf.nn.softmax(scaled_score, axis=-1)
        weights = self.dropout_layer(weights)
        
        output = tf.matmul(weights, value)
        return output, weights
    
    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs, mask=None, training=False):
        batch_size = tf.shape(inputs)[0]
        
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        
        attention, weights = self.attention(query, key, value, mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        
        output = self.combine_heads(concat_attention)
        return output


class TransformerBlock(layers.Layer):
    """Transformer block with multi-head attention and feed-forward network"""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
    
    def call(self, inputs, mask=None, training=False):
        attn_output = self.att(inputs, mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class FantasyFootballTransformer(Model):
    """
    Transformer-based model for fantasy football prediction
    Captures complex interactions between features
    """
    
    def __init__(self, 
                 num_features: int,
                 num_players: int = 1000,
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 ff_dim: int = 512,
                 num_transformer_blocks: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_features = num_features
        self.embed_dim = embed_dim
        
        # Input processing
        self.feature_embedding = layers.Dense(embed_dim)
        self.player_embedding = layers.Embedding(num_players, embed_dim)
        self.position_encoding = self._get_positional_encoding(1000, embed_dim)
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_transformer_blocks)
        ]
        
        # Output layers
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dropout = layers.Dropout(dropout)
        self.output_dense = keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            layers.Dense(1)  # Fantasy points prediction
        ])
    
    def _get_positional_encoding(self, max_len, d_model):
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def call(self, inputs, training=False):
        # Expect inputs to be (batch_size, sequence_length, num_features)
        seq_len = tf.shape(inputs)[1]
        
        # Embed features
        x = self.feature_embedding(inputs)
        
        # Add positional encoding
        x += self.position_encoding[:, :seq_len, :]
        
        # Pass through transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x, training=training)
        
        # Global pooling and output
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        output = self.output_dense(x)
        
        return output


class FantasyFootballLSTM(Model):
    """
    LSTM-based model for sequential fantasy football prediction
    Captures temporal dependencies in player performance
    """
    
    def __init__(self,
                 num_features: int,
                 lstm_units: List[int] = [128, 64, 32],
                 dense_units: List[int] = [256, 128, 64],
                 dropout: float = 0.3,
                 recurrent_dropout: float = 0.2):
        super().__init__()
        
        self.num_features = num_features
        
        # LSTM layers
        self.lstm_layers = []
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            self.lstm_layers.append(
                layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=keras.regularizers.l2(0.01)
                )
            )
        
        # Attention mechanism
        self.attention = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=32,
            dropout=dropout
        )
        
        # Dense layers
        self.dense_layers = []
        for units in dense_units:
            self.dense_layers.extend([
                layers.Dense(units, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(dropout)
            ])
        
        # Output layer
        self.output_layer = layers.Dense(1)
        
    def call(self, inputs, training=False):
        # Apply LSTM layers
        x = inputs
        for lstm in self.lstm_layers[:-1]:
            x = lstm(x, training=training)
        
        # Apply attention before final LSTM
        attended = self.attention(x, x, training=training)
        x = layers.Add()([x, attended])
        
        # Final LSTM
        x = self.lstm_layers[-1](x, training=training)
        
        # Dense layers
        for layer in self.dense_layers:
            x = layer(x, training=training)
        
        # Output
        output = self.output_layer(x)
        return output


class FantasyFootballCNN(Model):
    """
    CNN-based model for pattern recognition in player statistics
    Treats player stats as multi-channel images
    """
    
    def __init__(self,
                 num_features: int,
                 num_weeks: int = 17,
                 filters: List[int] = [64, 128, 256],
                 kernel_sizes: List[int] = [3, 3, 3],
                 dropout: float = 0.3):
        super().__init__()
        
        self.num_features = num_features
        self.num_weeks = num_weeks
        
        # Reshape layer to create 2D input
        self.reshape = layers.Reshape((num_weeks, num_features, 1))
        
        # Convolutional blocks
        self.conv_blocks = []
        for i, (filters, kernel_size) in enumerate(zip(filters, kernel_sizes)):
            self.conv_blocks.append(keras.Sequential([
                layers.Conv2D(filters, (kernel_size, kernel_size), 
                            padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 1)),
                layers.Dropout(dropout)
            ]))
        
        # Global pooling
        self.global_pool = layers.GlobalAveragePooling2D()
        
        # Dense layers
        self.dense_layers = keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            layers.Dense(1)
        ])
    
    def call(self, inputs, training=False):
        # Reshape to 2D
        x = self.reshape(inputs)
        
        # Apply conv blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x, training=training)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Dense layers and output
        output = self.dense_layers(x, training=training)
        return output


class HybridFantasyModel(Model):
    """
    Hybrid model combining CNN, LSTM, and Transformer architectures
    Leverages strengths of each architecture
    """
    
    def __init__(self,
                 num_features: int,
                 num_static_features: int,
                 sequence_length: int = 10,
                 cnn_filters: List[int] = [32, 64],
                 lstm_units: int = 64,
                 transformer_heads: int = 4,
                 embed_dim: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        
        # CNN branch for pattern extraction
        self.cnn_branch = keras.Sequential([
            layers.Conv1D(cnn_filters[0], 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Conv1D(cnn_filters[1], 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D()
        ])
        
        # LSTM branch for sequential patterns
        self.lstm_branch = keras.Sequential([
            layers.LSTM(lstm_units, return_sequences=True, dropout=dropout),
            layers.LSTM(lstm_units // 2, dropout=dropout)
        ])
        
        # Transformer branch for attention-based patterns
        self.transformer_branch = keras.Sequential([
            TransformerBlock(embed_dim, transformer_heads, embed_dim * 4, dropout),
            layers.GlobalAveragePooling1D()
        ])
        
        # Static features processing
        self.static_processor = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout)
        ])
        
        # Feature embedding for transformer
        self.feature_embedding = layers.Dense(embed_dim)
        
        # Fusion layer
        self.fusion = keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout)
        ])
        
        # Output layer
        self.output_layer = layers.Dense(1)
    
    def call(self, inputs, training=False):
        # Expect inputs to be a dictionary with 'sequential' and 'static' keys
        sequential_features = inputs['sequential']
        static_features = inputs['static']
        
        # Process through each branch
        cnn_features = self.cnn_branch(sequential_features, training=training)
        lstm_features = self.lstm_branch(sequential_features, training=training)
        
        # Prepare for transformer
        embedded_features = self.feature_embedding(sequential_features)
        transformer_features = self.transformer_branch(embedded_features, training=training)
        
        # Process static features
        static_processed = self.static_processor(static_features, training=training)
        
        # Concatenate all features
        combined_features = layers.Concatenate()([
            cnn_features, lstm_features, transformer_features, static_processed
        ])
        
        # Fusion and output
        fused = self.fusion(combined_features, training=training)
        output = self.output_layer(fused)
        
        return output


class AttentionWeightedEnsemble(Model):
    """
    Attention-based ensemble that learns to weight different model predictions
    """
    
    def __init__(self, num_models: int, hidden_dim: int = 64):
        super().__init__()
        
        self.num_models = num_models
        
        # Attention mechanism to weight models
        self.attention_dense = keras.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(num_models, activation='softmax')
        ])
        
        # Feature processor
        self.feature_processor = keras.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(hidden_dim // 2, activation='relu')
        ])
        
        # Final prediction layer
        self.output_layer = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
    
    def call(self, inputs, training=False):
        # Expect inputs to be {'predictions': [n_models, batch_size, 1], 'features': [batch_size, n_features]}
        predictions = inputs['predictions']  # Shape: (batch_size, n_models)
        features = inputs['features']  # Shape: (batch_size, n_features)
        
        # Calculate attention weights based on features
        processed_features = self.feature_processor(features, training=training)
        attention_weights = self.attention_dense(processed_features)
        
        # Apply attention weights to predictions
        weighted_predictions = tf.reduce_sum(
            predictions * tf.expand_dims(attention_weights, -1), 
            axis=1
        )
        
        # Combine weighted predictions with processed features
        combined = layers.Concatenate()([weighted_predictions, processed_features])
        
        # Final output
        output = self.output_layer(combined, training=training)
        return output