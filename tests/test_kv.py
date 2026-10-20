from gateway.kv import kv_bytes_per_token, reservation_tokens, round_up_to_blocks


def test_qwen3_8b_bf16_is_144_kib_per_token():
    # config.json: num_hidden_layers=36, num_key_value_heads=8, head_dim=128; BF16 = 2 bytes
    assert kv_bytes_per_token(36, 8, 128, 2) == 144 * 1024


def test_gqa_vs_mha_uses_kv_heads_not_query_heads():
    mha = kv_bytes_per_token(32, 32, 128, 2)  # 32 KV heads
    gqa = kv_bytes_per_token(32, 8, 128, 2)  # 8 KV heads, same model otherwise
    assert mha == 4 * gqa


def test_round_up_to_blocks():
    assert round_up_to_blocks(0, 16) == 0
    assert round_up_to_blocks(16, 16) == 16
    assert round_up_to_blocks(17, 16) == 32


def test_reservation_covers_prompt_and_output_in_whole_blocks():
    assert reservation_tokens(4000, 300, 16) == 4304
    assert reservation_tokens(4005, 300, 16) == 4320
