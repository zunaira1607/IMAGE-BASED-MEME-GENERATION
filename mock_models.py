"""
Simplified mock models for the meme generator
"""

class MockModel:
    def __init__(self, model_type="LSTM", tokenizer_type="char"):
        self.model_type = model_type
        self.tokenizer_type = tokenizer_type
        self.device = "cpu"
    
    def to(self, device):
        # Mock method
        return self

class LSTMMemeCaptioner(MockModel):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, device, num_layers=2, dropout=0.1):
        super().__init__(model_type="LSTM", tokenizer_type="char" if vocab_size < 200 else "word")
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.device = device

class TransformerMemeCaptioner(MockModel):
    def __init__(self, embedding_dim, num_heads, num_layers, vocab_size, device, dropout=0.1):
        super().__init__(model_type="Transformer", tokenizer_type="char" if vocab_size < 200 else "word")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.device = device