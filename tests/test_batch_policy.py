from pathlib import Path
import sys

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from _common import resolve_dataset_fixed_batch_policy, resolve_learning_rate_policy


def test_dataset_fixed_batch_policy_small_datasets():
    policy = resolve_dataset_fixed_batch_policy(train_n=21, input_dim=2, seq_len=640)
    assert policy["initial_batch_size"] == 4
    assert policy["batch_size"] == 3
    assert policy["singleton_remainder_adjusted"] is True

    policy = resolve_dataset_fixed_batch_policy(train_n=28, input_dim=1, seq_len=512)
    assert policy["batch_size"] == 4
    assert policy["train_steps_per_epoch"] == 7
    assert policy["dropped_train_samples_per_epoch"] == 0


def test_dataset_fixed_batch_policy_target_steps():
    policy = resolve_dataset_fixed_batch_policy(train_n=251, input_dim=6, seq_len=204)
    assert policy["batch_size"] == 16
    assert policy["train_steps_per_epoch"] == 16
    assert policy["drop_last"] is False
    assert policy["dropped_train_samples_per_epoch"] == 0

    assert resolve_dataset_fixed_batch_policy(train_n=448, input_dim=12, seq_len=29)["batch_size"] == 32


def test_dataset_fixed_batch_policy_footprint_caps_singleton_and_tiny_train():
    assert resolve_dataset_fixed_batch_policy(train_n=1000, input_dim=500, seq_len=200)["batch_size"] == 16
    assert resolve_dataset_fixed_batch_policy(train_n=1000, input_dim=1000, seq_len=300)["batch_size"] == 8

    singleton = resolve_dataset_fixed_batch_policy(train_n=33, input_dim=1, seq_len=20)
    assert singleton["initial_batch_size"] == 4
    assert singleton["batch_size"] == 5
    assert singleton["singleton_remainder_adjusted"] is True
    assert singleton["last_batch_size"] == 3
    assert singleton["singleton_last_batch"] is False

    policy = resolve_dataset_fixed_batch_policy(train_n=2, input_dim=1, seq_len=20)
    assert policy["batch_size"] == 2
    assert policy["train_steps_per_epoch"] == 1


def test_learning_rate_policy_fixed_fallback_and_config_passthrough():
    fallback = resolve_learning_rate_policy(None)
    assert fallback["source"] == "fixed_fallback"
    assert fallback["lr"] == 1e-3
    assert fallback["batch_size_coupled"] is False
    assert abs(fallback["weight_decay"] - 1e-5) < 1e-12

    configured = resolve_learning_rate_policy(3e-4)
    assert configured["source"] == "config"
    assert configured["lr"] == 3e-4
    assert abs(configured["weight_decay"] - 3e-6) < 1e-12
