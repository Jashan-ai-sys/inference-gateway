"""KV-cache arithmetic used for admission control and documentation."""


def kv_bytes_per_token(
    num_hidden_layers: int, num_key_value_heads: int, head_dim: int, dtype_bytes: int
) -> int:
    """Keys and values (x2) for every layer. Uses KV heads, not attention heads: under GQA the
    KV cache is smaller than the query-head count suggests."""
    return 2 * num_hidden_layers * num_key_value_heads * head_dim * dtype_bytes


def round_up_to_blocks(tokens: int, block_size: int) -> int:
    """vLLM allocates the KV cache in fixed-size blocks, so a partial block costs a whole one."""
    return -(-tokens // block_size) * block_size


def reservation_tokens(prompt_tokens: int, output_estimate: int, block_size: int) -> int:
    return round_up_to_blocks(prompt_tokens + output_estimate, block_size)
