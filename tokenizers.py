import re
import torch
from collections import Counter

class CharacterTokenizer:
    """
    Character-level tokenizer for meme captions.
    Maps individual characters to token IDs and vice versa.
    """
    def __init__(self):
        # Default character vocabulary
        # We're using a simple ASCII character set plus common punctuation
        self.chars = [chr(i) for i in range(32, 127)]  # printable ASCII
        self.char_to_id = {c: i+1 for i, c in enumerate(self.chars)}  # Start from 1, 0 is reserved for padding
        self.id_to_char = {i+1: c for i, c in enumerate(self.chars)}
        
        # Special tokens
        self.pad_token_id = 0
        self.unk_token_id = len(self.chars) + 1
        self.start_token_id = len(self.chars) + 2
        self.end_token_id = len(self.chars) + 3
        
        # Update dictionaries with special tokens
        self.id_to_char[self.pad_token_id] = '<PAD>'
        self.id_to_char[self.unk_token_id] = '<UNK>'
        self.id_to_char[self.start_token_id] = '<START>'
        self.id_to_char[self.end_token_id] = '<END>'
        
        self.char_to_id['<PAD>'] = self.pad_token_id
        self.char_to_id['<UNK>'] = self.unk_token_id
        self.char_to_id['<START>'] = self.start_token_id
        self.char_to_id['<END>'] = self.end_token_id
        
        self.vocab_size = len(self.char_to_id) + 1  # +1 for any characters not in the set
    
    def encode(self, text):
        """Convert text to a list of token IDs"""
        return [self.start_token_id] + [self.char_to_id.get(c, self.unk_token_id) for c in text] + [self.end_token_id]
    
    def decode(self, tokens):
        """Convert a list of token IDs back to text"""
        result = []
        for t in tokens:
            if t == self.start_token_id or t == self.pad_token_id:
                continue
            if t == self.end_token_id:
                break
            result.append(self.id_to_char.get(t, '<UNK>'))
        return ''.join(result)
    
    def batch_encode(self, texts, max_length=None):
        """Encode a batch of texts, padding to max_length if specified"""
        encoded = [self.encode(text) for text in texts]
        
        if max_length is not None:
            # Pad or truncate to max_length
            encoded = [seq[:max_length-1] + [self.end_token_id] if len(seq) > max_length else 
                      seq + [self.pad_token_id] * (max_length - len(seq)) for seq in encoded]
        
        return torch.tensor(encoded)


class WordTokenizer:
    """
    Word-level tokenizer for meme captions.
    Maps words to token IDs and vice versa.
    """
    def __init__(self, vocab_size=10000):
        self.word_to_id = {}
        self.id_to_word = {}
        self.max_vocab_size = vocab_size
        
        # Special tokens
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.start_token_id = 2
        self.end_token_id = 3
        
        self.word_to_id['<PAD>'] = self.pad_token_id
        self.word_to_id['<UNK>'] = self.unk_token_id
        self.word_to_id['<START>'] = self.start_token_id
        self.word_to_id['<END>'] = self.end_token_id
        
        self.id_to_word[self.pad_token_id] = '<PAD>'
        self.id_to_word[self.unk_token_id] = '<UNK>'
        self.id_to_word[self.start_token_id] = '<START>'
        self.id_to_word[self.end_token_id] = '<END>'
        
        self.vocab_size = len(self.word_to_id)
        
        # Default to a small set of common words initially
        self._initialize_default_vocab()
    
    def _initialize_default_vocab(self):
        """
        Initialize with some common English words.
        In a real implementation, this would be trained on a corpus of meme captions.
        """
        common_words = [
            "the", "and", "to", "of", "a", "in", "that", "is", "was", "for",
            "it", "with", "be", "as", "on", "at", "by", "this", "an", "but",
            "not", "are", "from", "or", "have", "you", "me", "my", "when", "what",
            "who", "how", "why", "where", "which", "if", "then", "now", "here", "there",
            "can", "will", "would", "should", "could", "just", "like", "so", "very", "too",
            "meme", "funny", "joke", "lol", "haha", "wow", "yes", "no", "man", "woman",
            "dog", "cat", "people", "time", "day", "life", "good", "bad", "happy", "sad"
        ]
        
        for word in common_words:
            if word not in self.word_to_id:
                self.word_to_id[word] = len(self.word_to_id)
                self.id_to_word[len(self.id_to_word)] = word
        
        self.vocab_size = len(self.word_to_id)
    
    def fit_on_texts(self, texts):
        """Build vocabulary from a list of texts"""
        word_counts = Counter()
        
        for text in texts:
            for word in self._tokenize(text):
                word_counts[word] += 1
        
        # Sort by frequency and take the most common
        most_common = word_counts.most_common(self.max_vocab_size - len(self.word_to_id))
        
        for word, _ in most_common:
            if word not in self.word_to_id:
                self.word_to_id[word] = len(self.word_to_id)
                self.id_to_word[len(self.id_to_word)] = word
        
        self.vocab_size = len(self.word_to_id)
    
    def _tokenize(self, text):
        """Simple word tokenization"""
        # Replace punctuation with spaces
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()
    
    def encode(self, text):
        """Convert text to a list of token IDs"""
        words = self._tokenize(text)
        return [self.start_token_id] + [self.word_to_id.get(word, self.unk_token_id) for word in words] + [self.end_token_id]
    
    def decode(self, tokens):
        """Convert a list of token IDs back to text"""
        result = []
        for t in tokens:
            if t == self.start_token_id or t == self.pad_token_id:
                continue
            if t == self.end_token_id:
                break
            result.append(self.id_to_word.get(t, '<UNK>'))
        return ' '.join(result)
    
    def batch_encode(self, texts, max_length=None):
        """Encode a batch of texts, padding to max_length if specified"""
        encoded = [self.encode(text) for text in texts]
        
        if max_length is not None:
            # Pad or truncate to max_length
            encoded = [seq[:max_length-1] + [self.end_token_id] if len(seq) > max_length else 
                      seq + [self.pad_token_id] * (max_length - len(seq)) for seq in encoded]
        
        return torch.tensor(encoded)
