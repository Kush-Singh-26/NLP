from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="Kush26/Transformer_Translation",
    filename="model.pth",
    local_dir="."
)

hf_hub_download(
    repo_id="Kush26/Transformer_Translation", 
    filename="hindi-english_bpe_tokenizer.json",
    local_dir="."
)