import torch
import torch.nn as nn
import math
import sys

from tokenizers import Tokenizer

MODEL_PATH = './model.pth'
TOKENIZER_PATH = './hindi-english_bpe_tokenizer.json'

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
vocab_size = tokenizer.get_vocab_size()
pad_token_id = tokenizer.token_to_id('[PAD]')

SOS_token = tokenizer.token_to_id('[SOS]')
EOS_token = tokenizer.token_to_id('[EOS]')
PAD_token = tokenizer.token_to_id('[PAD]')


class InputEmbedding(nn.Module):

  def __init__(self, d_model, vocab_size):
    super().__init__()
    self.d_model = d_model
    self.vocab_size = vocab_size
    self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

  def forward(self, x):

    return self.embed(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

  def __init__(self, d_model, seq_len, dropout):
    super().__init__()
    self.d_model = d_model
    self.seq_len = seq_len
    self.dropout = nn.Dropout(dropout)
    pe = torch.zeros(seq_len, d_model) # matrix of shape same as embedings
    pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # tensor of shape [seq_len, 1] denotes the position of token
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # shape of tensor div_term = [d_model // 2]
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    pe = pe.unsqueeze(0) # shape of pe = [1, seq_len, d_model]

    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)  # slicing is done to avoid shape mismatch in variable length sequence
    return self.dropout(x)
  
class LayerNorm(nn.Module):

  def __init__(self, d_model, epsilon = 10**-6):

    super().__init__()
    self.epsilon = epsilon
    self.gamma = nn.Parameter(torch.ones(d_model))
    self.beta = nn.Parameter(torch.zeros(d_model))

  # x shape = [batch_size, seq_len, d_model]
  def forward(self, x):

    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)

    return self.gamma * (x - mean) / (std + self.epsilon) + self.beta # mathematically not exact

class FeedForward(nn.Module):

  def __init__(self, d_model, d_ff, dropout):

    super().__init__()
    self.layer1 = nn.Linear(d_model, d_ff)
    self.layer2 = nn.Linear(d_ff, d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):

    return self.layer2(self.dropout(torch.relu(self.layer1(x))))
  
class MHA(nn.Module):

  def __init__(self, d_model, h, dropout):

    super().__init__()
    self.d_model = d_model
    self.h = h
    self.dropout = nn.Dropout(dropout)

    self.d_k = d_model // h # d_k = d_v
    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)

    self.w_o = nn.Linear(d_model, d_model)

  def forward(self, q, k, v, mask):

    batch_size, seq_len, _ = q.size()

    query = self.w_q(q) # shape of both query and key = [batch_size, seq_len, d_model]
    key = self.w_k(k) # same as query
    value = self.w_v(v) # same as query

    query = query.view(batch_size, -1, self.h, self.d_k) # shape = [batch_size, seq_len, h, d_k]
    query = query.transpose(1, 2) # shape = [batch_size, h, seq_len, d_k]
    key = key.view(batch_size, -1, self.h, self.d_k)
    key = key.transpose(1, 2)
    value = value.view(batch_size, -1, self.h, self.d_k)
    value = value.transpose(1, 2)

    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k) # shape = [batch_size, h, seq_len, seq_len]

    if mask is not None:
      attention_scores = attention_scores.masked_fill_(mask == 0, float('-inf'))

    attention_weights = attention_scores.softmax(dim=-1)

    if self.dropout is not None:
      attention_weights = self.dropout(attention_weights)

    attention_output = attention_weights @  value # shape = [batch_size, h, seq_len, d_k]

    attention_output = attention_output.transpose(1, 2) # shape = [batch_size, seq_len, h, d_k]
    attention_output = attention_output.contiguous() # makes the tensor contiguous in memory for .view as transpose may result in tensor not being stored in a contiguous block of memory
    attention_output = attention_output.view(batch_size, seq_len, self.d_model) # shape = [batch_size, seq_len, d_model]
    attention_output = self.w_o(attention_output) # final projection, same shape
    return attention_output

class SkipConnection(nn.Module):

  def __init__(self, dropout, d_model):

    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.norm = LayerNorm(d_model)

  def forward(self, x, sublayer):

    return x + self.dropout(sublayer(self.norm(x))) # pre-norm

class EncoderBlock(nn.Module):

  def __init__(self, attention, ffn, dropout, d_model):

    super().__init__()
    self.attention = attention
    self.ffn = ffn
    self.residual = nn.ModuleList([SkipConnection(dropout, d_model) for _ in range(2)])

  # src_mask is used to mask out padding tokens in encoder
  def forward(self, x, src_mask):
    x = self.residual[0](x, lambda y: self.attention(y, y, y, src_mask))
    x = self.residual[1](x, self.ffn)
    return x

class Encoder(nn.Module):

  def __init__(self, d_model, layers):

    super().__init__()
    self.layers = layers
    self.norm = LayerNorm(d_model)

  def forward(self, x, mask):

    for layer in self.layers:
      x = layer(x, mask)
    return self.norm(x)

class DecoderBlock(nn.Module):

  def __init__(self, self_attention, cross_attention, ffn, dropout, d_model):

    super().__init__()
    self.self_attention = self_attention
    self.cross_attention = cross_attention
    self.ffn = ffn
    self.residual = nn.ModuleList([SkipConnection(dropout, d_model) for _ in range(3)])

  def forward(self, x, encoder_output, src_mask, trg_mask):

    x = self.residual[0](x, lambda y: self.self_attention(y, y, y, trg_mask))
    x = self.residual[1](x, lambda y: self.cross_attention(y, encoder_output, encoder_output, src_mask))
    x = self.residual[2](x, self.ffn)

    return x

class Decoder(nn.Module):

  def __init__(self, d_model, layers):

    super().__init__()
    self.layers = layers
    self.norm = LayerNorm(d_model)

  def forward(self, x, encoder_output, src_mask, trg_mask):

    for layer in self.layers:
      x = layer(x, encoder_output, src_mask, trg_mask)

    return self.norm(x)

class Output(nn.Module):

  def __init__(self, d_model, vocab_size):

    super().__init__()
    self.proj = nn.Linear(d_model, vocab_size)

  def forward(self, x):

    return self.proj(x)

class Transformer(nn.Module):

  def __init__(self, encoder, decoder, src_embed, trg_embed, src_pos, trg_pos, output):

    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_embed = src_embed
    self.trg_embed = trg_embed
    self.src_pos = src_pos
    self.trg_pos = trg_pos
    self.output_layer = output

  def encode(self, src, src_mask):

    src = self.src_embed(src)
    src = self.src_pos(src)
    return self.encoder(src, src_mask)

  def decode(self, encoder_output, src_mask, trg, trg_mask):

    trg = self.trg_embed(trg)
    trg = self.trg_pos(trg)
    return self.decoder(trg, encoder_output, src_mask, trg_mask)

  def project(self, x):

    return self.output_layer(x)

  def forward(self, src, trg):
        # Create masks for source and target
        # Target mask is a combination of padding mask and subsequent mask
        src_mask = (src != PAD_token).unsqueeze(1).unsqueeze(2) # (batch, 1, 1, src_len)
        trg_mask = (trg != PAD_token).unsqueeze(1).unsqueeze(2) # (batch, 1, 1, trg_len)

        seq_length = trg.size(1)
        subsequent_mask = torch.tril(torch.ones(1, seq_length, seq_length)).to(device) # (1, trg_len, trg_len)
        trg_mask = trg_mask & (subsequent_mask==1)

        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(encoder_output, src_mask, trg, trg_mask)
        return self.project(decoder_output)

def BuildTransformer(src_vocab_size, trg_vocab_size, src_seq_len, trg_seq_len, d_model=512, N=6, h=8, dropout=0.1, d_ff=2048):

  src_embed = InputEmbedding(d_model, src_vocab_size)
  trg_embed = InputEmbedding(d_model, trg_vocab_size)

  src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
  trg_pos = PositionalEncoding(d_model, trg_seq_len, dropout)

  encoder_blocks = []
  for _ in range(N):
    encoder_self_attention = MHA(d_model, h, dropout)
    ffn = FeedForward(d_model, d_ff, dropout)
    encoder_block = EncoderBlock(encoder_self_attention, ffn, dropout, d_model)
    encoder_blocks.append(encoder_block)

  decoder_blocks = []
  for _ in range(N):
    decoder_mask_attention = MHA(d_model, h, dropout)
    cross_attention = MHA(d_model, h, dropout)
    ffn = FeedForward(d_model, d_ff, dropout)
    decoder_block = DecoderBlock(decoder_mask_attention, cross_attention, ffn, dropout, d_model)
    decoder_blocks.append(decoder_block)

  encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
  decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

  projection = Output(d_model, trg_vocab_size)

  transformer = Transformer(encoder, decoder, src_embed, trg_embed, src_pos, trg_pos, projection)

  for p in transformer.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)

  return transformer

config = {
    "d_model": 256,
    "num_layers": 6,
    "num_heads": 8,
    "d_ff": 2048,
    "dropout": 0.1,
    "max_seq_len": 512,
}

device = torch.device("cpu")

model = BuildTransformer(vocab_size,
                         vocab_size,
                         config["max_seq_len"],
                         config["max_seq_len"],
                         config["d_model"],
                         config["num_layers"],
                         config["num_heads"],
                         config["dropout"],
                         config["d_ff"]).to(device)

# total_parameters = sum(p.numel() for p in model.parameters())
# print(f"Totoal Parameters = {total_parameters}")

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def translate_sentence(sentence: str, model, tokenizer, device, max_len=100):
    model.eval()

    src_ids = [tokenizer.token_to_id('[SOS]')] + tokenizer.encode(sentence).ids + [tokenizer.token_to_id('[EOS]')]
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)
    src_mask = (src_tensor != PAD_token).unsqueeze(1).unsqueeze(2)

    with torch.no_grad():
        encoder_output = model.encode(src_tensor, src_mask)

    tgt_tokens = [tokenizer.token_to_id('[SOS]')]
    
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(device)
        
        trg_mask_padding = (tgt_tensor != PAD_token).unsqueeze(1).unsqueeze(2)
        subsequent_mask = torch.tril(torch.ones(1, tgt_tensor.size(1), tgt_tensor.size(1))).to(device)
        trg_mask = trg_mask_padding & (subsequent_mask == 1)

        with torch.no_grad():
            decoder_output = model.decode(encoder_output, src_mask, tgt_tensor, trg_mask)
            logits = model.project(decoder_output)
        
        pred_token = logits.argmax(dim=-1)[0, -1].item()
        
        tgt_tokens.append(pred_token)
        
        if pred_token == tokenizer.token_to_id('[EOS]'):
            break
            
    translated_text = tokenizer.decode(tgt_tokens, skip_special_tokens=True)
    
    return translated_text

import torch.nn.functional as F 

def translate_beam_search(sentence, model, tokenizer, device, pad_token_id, beam_size=3, max_len=50):

    model.eval()

    src_ids = [tokenizer.token_to_id('[SOS]')] + tokenizer.encode(sentence).ids + [tokenizer.token_to_id('[EOS]')]
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)
    src_mask = (src_tensor != pad_token_id).unsqueeze(1).unsqueeze(2)

    with torch.no_grad():
        encoder_output = model.encode(src_tensor, src_mask)

    initial_beam = (torch.tensor([tokenizer.token_to_id('[SOS]')], device=device), 0.0)
    beams = [initial_beam]

    for _ in range(max_len):
        new_beams = []
        
        for seq, score in beams:
            if seq[-1].item() == tokenizer.token_to_id('[EOS]'):
                new_beams.append((seq, score))
                continue

            tgt_tensor = seq.unsqueeze(0)
            trg_mask_padding = (tgt_tensor != pad_token_id).unsqueeze(1).unsqueeze(2)
            subsequent_mask = torch.tril(torch.ones(1, tgt_tensor.size(1), tgt_tensor.size(1))).to(device)
            trg_mask = trg_mask_padding & (subsequent_mask == 1)

            with torch.no_grad():
                decoder_output = model.decode(encoder_output, src_mask, tgt_tensor, trg_mask)
                logits = model.project(decoder_output)
            
            last_token_logits = logits[0, -1, :]
            log_probs = F.log_softmax(last_token_logits, dim=-1)

            top_log_probs, top_next_tokens = torch.topk(log_probs, beam_size)

            for i in range(beam_size):
                next_token = top_next_tokens[i]
                log_prob = top_log_probs[i].item()

                new_seq = torch.cat([seq, next_token.unsqueeze(0)])
                new_score = score + log_prob
                
                new_beams.append((new_seq, new_score))
        
        new_beams.sort(key=lambda x: x[1], reverse=True)
        
        beams = new_beams[:beam_size]
        
        if beams[0][0][-1].item() == tokenizer.token_to_id('[EOS]'):
            break

    best_seq = beams[0][0]
    
    return tokenizer.decode(best_seq.tolist(), skip_special_tokens=True)


if len(sys.argv) > 2:
        mode = sys.argv[1]
        if mode == "-g" : 
          sentence_to_translate = " ".join(sys.argv[2:]) 
          translation = translate_sentence(sentence_to_translate, model, tokenizer, device, 100)
        elif mode == '-b':
          sentence_to_translate = " ".join(sys.argv[3:]) 
          beam_size = int(sys.argv[2])
          translation = translate_beam_search(sentence_to_translate, model, tokenizer, device, 100, beam_size=beam_size)
        print(f"Hindi Translation: {translation}")
else:
        print("\nUsage:\n  python inference.py -g <sentence>\n  python inference.py -b <beam_size> <sentence>")