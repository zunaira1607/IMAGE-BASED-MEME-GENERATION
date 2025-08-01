import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LSTMMemeCaptioner(nn.Module):
    """
    LSTM-based model for meme caption generation as described in the paper.
    Uses an image encoder and LSTM decoder architecture.
    """
    def __init__(self, embedding_dim, hidden_dim, vocab_size, device, num_layers=2, dropout=0.1):
        super(LSTMMemeCaptioner, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.device = device
        
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Image feature projection
        self.img_projection = nn.Linear(2048, embedding_dim)  # ResNet features are 2048-dim
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize to device
        self.to(device)
    
    def forward(self, image_features, captions=None, max_len=50):
        """
        Forward pass for either training or inference
        Args:
            image_features: Tensor of shape [batch_size, 2048]
            captions: Tensor of shape [batch_size, seq_len] or None for inference
            max_len: Maximum length of generated caption during inference
        """
        batch_size = image_features.size(0)
        
        # Project image features
        projected_img = self.img_projection(image_features)
        
        # Initialize hidden state with image features
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        c0 = projected_img.unsqueeze(0).repeat(self.num_layers, 1, 1)
        hidden = (h0, c0)
        
        if captions is not None:
            # Training mode - teacher forcing
            embedded = self.embedding(captions)
            output, _ = self.lstm(embedded, hidden)
            logits = self.output_projection(output)
            return logits
        else:
            # Inference mode
            current_token = torch.ones(batch_size, 1).long().to(self.device)  # Start token
            generated_tokens = []
            
            for i in range(max_len):
                embedded = self.embedding(current_token)
                output, hidden = self.lstm(embedded, hidden)
                logits = self.output_projection(output)
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                current_token = torch.argmax(probs, dim=-1)
                
                generated_tokens.append(current_token)
                
            return torch.cat(generated_tokens, dim=1)


class TransformerMemeCaptioner(nn.Module):
    """
    Transformer-based model for meme caption generation as described in the paper.
    Uses self-attention mechanism with encoder-decoder architecture.
    """
    def __init__(self, embedding_dim, num_heads, num_layers, vocab_size, device, dropout=0.1):
        super(TransformerMemeCaptioner, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.device = device
        
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        
        # Image feature projection
        self.img_projection = nn.Linear(2048, embedding_dim)  # ResNet features are 2048-dim
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        
        # Initialize to device
        self.to(device)
    
    def forward(self, image_features, captions=None, max_len=50):
        """
        Forward pass for either training or inference
        Args:
            image_features: Tensor of shape [batch_size, 2048]
            captions: Tensor of shape [batch_size, seq_len] or None for inference
            max_len: Maximum length of generated caption during inference
        """
        batch_size = image_features.size(0)
        
        # Project and prepare image features for transformer
        memory = self.img_projection(image_features).unsqueeze(1)  # [batch_size, 1, embedding_dim]
        memory = memory.permute(1, 0, 2)  # [1, batch_size, embedding_dim]
        
        if captions is not None:
            # Training mode - with teacher forcing
            tgt = self.embedding(captions)
            tgt = self.pos_encoder(tgt.permute(1, 0, 2))  # [seq_len, batch_size, embedding_dim]
            
            # Create causal mask for transformer
            tgt_mask = self._generate_square_subsequent_mask(captions.size(1)).to(self.device)
            
            # Transformer decoding
            output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
            output = output.permute(1, 0, 2)  # [batch_size, seq_len, embedding_dim]
            
            # Project to vocabulary
            logits = self.output_projection(output)
            return logits
        else:
            # Inference mode
            current_tokens = torch.ones(batch_size, 1).long().to(self.device)  # Start token
            generated_tokens = [current_tokens]
            
            for i in range(max_len - 1):
                tgt = self.embedding(torch.cat(generated_tokens, dim=1))
                tgt = self.pos_encoder(tgt.permute(1, 0, 2))
                
                tgt_mask = self._generate_square_subsequent_mask(tgt.size(0)).to(self.device)
                
                output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
                output = output.permute(1, 0, 2)  # [batch_size, seq_len, embedding_dim]
                
                # Get prediction for next token only (last position)
                next_token_logits = self.output_projection(output[:, -1:, :])
                next_token = torch.argmax(F.softmax(next_token_logits, dim=-1), dim=-1)
                
                generated_tokens.append(next_token)
                
            return torch.cat(generated_tokens, dim=1)
    
    def _generate_square_subsequent_mask(self, sz):
        """Generate causal mask for transformer decoder."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer model.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
