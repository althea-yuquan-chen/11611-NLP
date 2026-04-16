"""Student starter code for Homework 4: simplified DPO trainer.

Fill in the TODOs. The public tests import these functions directly.
Do not change function signatures.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import torch
import torch.nn.functional as F


IGNORE_INDEX = -100


def format_prompt(prompt: str) -> str:
    """Format a raw prompt for a plain causal LM."""
    prompt = prompt.strip()
    return f"User: {prompt}\nAssistant: "


def _tokenize_text(tokenizer: Any, text: str) -> List[int]:
    """Small compatibility wrapper for Hugging Face tokenizers and test tokenizers."""
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    encoded = tokenizer(text, add_special_tokens=False)
    if "input_ids" not in encoded:
        raise KeyError("Tokenizer output must contain 'input_ids'.")
    return list(encoded["input_ids"])


def build_lm_sequence(
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    *,
    eos_token_id: int,
    max_length: int,
    ignore_index: int = IGNORE_INDEX,
) -> Tuple[List[int], List[int]]:
    """Build a single prompt+response causal-LM example.

    The returned labels must mask out prompt tokens with `ignore_index`.
    The response should always end with EOS.
    If the combined example is too long, preserve as much of the response as possible
    and truncate the prompt from the left.

    Args:
        prompt_ids: token IDs for the prompt.
        response_ids: token IDs for the response (without EOS).
        eos_token_id: end-of-sequence token ID.
        max_length: maximum total length after appending EOS.
        ignore_index: label value used to mask prompt tokens.

    Returns:
        input_ids: concatenated prompt + response + EOS token IDs.
        labels: same length as input_ids, but prompt positions are ignore_index.
    """
    # Always append EOS to the response
    response_with_eos = list(response_ids) + [eos_token_id]

    # How many tokens are left for the prompt after fitting response + EOS
    prompt_budget = max_length - len(response_with_eos)

    if prompt_budget <= 0:
        # No room for prompt; just use the response (truncated if needed)
        prompt_ids_final: List[int] = []
    else:
        # Truncate prompt from the left to fit within budget
        prompt_ids_final = list(prompt_ids)[-prompt_budget:]

    input_ids = prompt_ids_final + response_with_eos
    labels = [ignore_index] * len(prompt_ids_final) + response_with_eos

    return input_ids, labels


def tokenize_preference_example(
    example: Mapping[str, str],
    tokenizer: Any,
    *,
    max_prompt_length: int,
    max_response_length: int,
    max_length: int,
    ignore_index: int = IGNORE_INDEX,
) -> Dict[str, List[int]]:
    """Tokenize one preference example.

    The input example has keys: "prompt", "chosen", "rejected".

    Expected output keys:
        - chosen_input_ids
        - chosen_labels
        - rejected_input_ids
        - rejected_labels

    Notes:
        * Prompt tokens should be truncated from the left.
        * Response tokens should be truncated from the right.
        * Reserve room for EOS in each response sequence.
    """
    formatted_prompt = format_prompt(example["prompt"])

    # Tokenize each field without special tokens
    prompt_ids = _tokenize_text(tokenizer, formatted_prompt)
    chosen_ids = _tokenize_text(tokenizer, example["chosen"])
    rejected_ids = _tokenize_text(tokenizer, example["rejected"])

    # Truncate prompt from the left
    prompt_ids = prompt_ids[-max_prompt_length:]

    # Truncate responses from the right; reserve 1 slot for EOS
    chosen_ids = chosen_ids[: max_response_length - 1]
    rejected_ids = rejected_ids[: max_response_length - 1]

    eos_token_id = tokenizer.eos_token_id

    chosen_input_ids, chosen_labels = build_lm_sequence(
        prompt_ids,
        chosen_ids,
        eos_token_id=eos_token_id,
        max_length=max_length,
        ignore_index=ignore_index,
    )
    rejected_input_ids, rejected_labels = build_lm_sequence(
        prompt_ids,
        rejected_ids,
        eos_token_id=eos_token_id,
        max_length=max_length,
        ignore_index=ignore_index,
    )

    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_input_ids,
        "rejected_labels": rejected_labels,
    }


def _pad_sequences(
    sequences: Sequence[Sequence[int]],
    *,
    pad_value: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of integer sequences to the same length.

    Args:
        sequences: list of variable-length token ID sequences.
        pad_value: value used for padding positions.

    Returns:
        padded: LongTensor of shape [batch, max_len].
        attention_mask: LongTensor of shape [batch, max_len], with 1 for real
            tokens and 0 for padding. Derived from original sequence lengths,
            so it is correct even when pad_value == eos_token_id.
     Important:
        Build attention masks from the original sequence lengths, not from token equality.
        This matters when pad_token_id == eos_token_id (as is common for GPT-2).
    """
    lengths = [len(s) for s in sequences]
    max_len = max(lengths)
    batch_size = len(sequences)

    padded = torch.full((batch_size, max_len), pad_value, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, (seq, length) in enumerate(zip(sequences, lengths)):
        padded[i, :length] = torch.tensor(list(seq), dtype=torch.long)
        attention_mask[i, :length] = 1

    return padded, attention_mask


def preference_collate_fn(
    examples: Sequence[Mapping[str, Sequence[int]]],
    *,
    pad_token_id: int,
    ignore_index: int = IGNORE_INDEX,
) -> Dict[str, torch.Tensor]:
    """Collate tokenized preference examples into a padded batch.

    The batch must contain:
        chosen_input_ids, chosen_labels, chosen_attention_mask,
        rejected_input_ids, rejected_labels, rejected_attention_mask

    Use _pad_sequences for padding. Call it separately for input_ids
    (pad with pad_token_id) and labels (pad with ignore_index) so that
    each field gets the correct pad value.
    """
    chosen_input_ids_list = [ex["chosen_input_ids"] for ex in examples]
    chosen_labels_list = [ex["chosen_labels"] for ex in examples]
    rejected_input_ids_list = [ex["rejected_input_ids"] for ex in examples]
    rejected_labels_list = [ex["rejected_labels"] for ex in examples]

    chosen_input_ids, chosen_attention_mask = _pad_sequences(
        chosen_input_ids_list, pad_value=pad_token_id
    )
    chosen_labels, _ = _pad_sequences(chosen_labels_list, pad_value=ignore_index)

    rejected_input_ids, rejected_attention_mask = _pad_sequences(
        rejected_input_ids_list, pad_value=pad_token_id
    )
    rejected_labels, _ = _pad_sequences(rejected_labels_list, pad_value=ignore_index)

    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_labels": chosen_labels,
        "chosen_attention_mask": chosen_attention_mask,
        "rejected_input_ids": rejected_input_ids,
        "rejected_labels": rejected_labels,
        "rejected_attention_mask": rejected_attention_mask,
    }


def sequence_logps_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
    average_log_prob: bool = False,
) -> torch.Tensor:
    """Compute sequence log-probabilities from causal-LM logits and labels.

    This function must:
        1. shift logits and labels for next-token prediction;
        2. ignore positions where labels == ignore_index;
        3. return one scalar log-probability per sequence.

    Args:
        logits: Float tensor of shape [batch, seq_len, vocab_size].
        labels: Long tensor of shape [batch, seq_len].
        ignore_index: masked label value.
        average_log_prob: if True, average over non-masked positions instead of summing.

    Returns:
        Tensor of shape [batch].
    """
    # Causal shift: logit at position t predicts token at position t+1
    shifted_logits = logits[:, :-1, :]  # [batch, seq_len-1, vocab]
    shifted_labels = labels[:, 1:]      # [batch, seq_len-1]

    # Compute log-softmax probabilities
    log_probs = F.log_softmax(shifted_logits, dim=-1)  # [batch, seq_len-1, vocab]

    # Mask for valid (non-ignored) positions
    mask = shifted_labels != ignore_index  # [batch, seq_len-1]

    # Replace ignore_index with 0 so gather doesn't fail on invalid indices
    gather_labels = shifted_labels.clone()
    gather_labels[~mask] = 0

    # Gather the log-prob of each target token: [batch, seq_len-1]
    token_logps = log_probs.gather(
        dim=-1, index=gather_labels.unsqueeze(-1)
    ).squeeze(-1)

    # Zero out ignored positions
    token_logps = token_logps * mask.float()

    if average_log_prob:
        seq_logps = token_logps.sum(dim=-1) / mask.float().sum(dim=-1).clamp(min=1)
    else:
        seq_logps = token_logps.sum(dim=-1)

    return seq_logps


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute per-example DPO losses.

    Returns:
        losses: Tensor of shape [batch].
        stats: dict containing detached tensors with at least:
            - chosen_rewards
            - rejected_rewards
            - margins
            - accuracy
    """
    # Implicit rewards: beta * log(pi_theta / pi_ref)
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    # Margin: difference in log-ratio between chosen and rejected
    margins = (policy_chosen_logps - policy_rejected_logps) - (
        ref_chosen_logps - ref_rejected_logps
    )

    # DPO loss: -log sigmoid(beta * margin)
    losses = -F.logsigmoid(beta * margins)

    stats = {
        "chosen_rewards": chosen_rewards.detach(),
        "rejected_rewards": rejected_rewards.detach(),
        "margins": margins.detach(),
        "accuracy": (margins > 0).float().detach(),
    }

    return losses, stats


def compute_dpo_batch(
    policy_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    beta: float,
    ignore_index: int = IGNORE_INDEX,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Run policy and reference models on one batch and return mean DPO loss + metrics.

    The returned scalar loss should require gradients with respect to the policy model.
    The reference model should be evaluated without tracking gradients.
    """
    chosen_input_ids = batch["chosen_input_ids"]
    chosen_labels = batch["chosen_labels"]
    chosen_attention_mask = batch["chosen_attention_mask"]
    rejected_input_ids = batch["rejected_input_ids"]
    rejected_labels = batch["rejected_labels"]
    rejected_attention_mask = batch["rejected_attention_mask"]

    # Policy forward passes — gradients flow here
    policy_chosen_logits = policy_model(
        chosen_input_ids, attention_mask=chosen_attention_mask
    ).logits
    policy_rejected_logits = policy_model(
        rejected_input_ids, attention_mask=rejected_attention_mask
    ).logits

    policy_chosen_logps = sequence_logps_from_logits(
        policy_chosen_logits, chosen_labels, ignore_index=ignore_index
    )
    policy_rejected_logps = sequence_logps_from_logits(
        policy_rejected_logits, rejected_labels, ignore_index=ignore_index
    )

    # Reference forward passes — no gradients
    with torch.no_grad():
        ref_chosen_logits = reference_model(
            chosen_input_ids, attention_mask=chosen_attention_mask
        ).logits
        ref_rejected_logits = reference_model(
            rejected_input_ids, attention_mask=rejected_attention_mask
        ).logits

    ref_chosen_logps = sequence_logps_from_logits(
        ref_chosen_logits, chosen_labels, ignore_index=ignore_index
    )
    ref_rejected_logps = sequence_logps_from_logits(
        ref_rejected_logits, rejected_labels, ignore_index=ignore_index
    )

    losses, stats = dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta,
    )

    mean_loss = losses.mean()

    metrics: Dict[str, torch.Tensor] = {
        "loss": mean_loss.detach(),
        "chosen_rewards": stats["chosen_rewards"].mean(),
        "rejected_rewards": stats["rejected_rewards"].mean(),
        "mean_margin": stats["margins"].mean(),
        "preference_accuracy": stats["accuracy"].mean(),
    }

    return mean_loss, metrics


def train_step(
    policy_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    beta: float,
    grad_clip_norm: float | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> Dict[str, float]:
    """Perform one optimization step on the policy model.

    Returns:
        A dictionary of Python floats, for example:
            {
                "loss": ...,
                "preference_accuracy": ...,
                "mean_margin": ...
            }
    """
    policy_model.train()
    optimizer.zero_grad()

    loss, metrics = compute_dpo_batch(
        policy_model,
        reference_model,
        batch,
        beta=beta,
        ignore_index=ignore_index,
    )

    loss.backward()

    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), grad_clip_norm)

    optimizer.step()

    return {
        "loss": loss.item(),
        "preference_accuracy": metrics["preference_accuracy"].item(),
        "mean_margin": metrics["mean_margin"].item(),
    }


@torch.no_grad()
def evaluate_preference_accuracy(
    policy_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    dataloader: Sequence[Mapping[str, torch.Tensor]],
    *,
    beta: float,
    ignore_index: int = IGNORE_INDEX,
) -> Dict[str, float]:
    """Evaluate a model on a preference dataloader.

    Aggregate metrics over all examples and return Python floats.
    Suggested keys:
        - loss
        - preference_accuracy
        - mean_margin
    """
    policy_model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    total_margin = 0.0
    n_batches = 0

    for batch in dataloader:
        loss, metrics = compute_dpo_batch(
            policy_model,
            reference_model,
            batch,
            beta=beta,
            ignore_index=ignore_index,
        )
        total_loss += loss.item()
        total_accuracy += metrics["preference_accuracy"].item()
        total_margin += metrics["mean_margin"].item()
        n_batches += 1

    if n_batches == 0:
        return {"loss": 0.0, "preference_accuracy": 0.0, "mean_margin": 0.0}

    return {
        "loss": total_loss / n_batches,
        "preference_accuracy": total_accuracy / n_batches,
        "mean_margin": total_margin / n_batches,
    }
