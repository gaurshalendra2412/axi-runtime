import torch
from typing import Dict, Set

class GrammarMaskLogitProcessor:
    def __init__(self, vocab_map: Dict[str, int]):
        self.vocab_map = vocab_map
        self.bracket_open = vocab_map.get("[", 1)
        self.bracket_close = vocab_map.get("]", 2)
        self.comma = vocab_map.get(",", 3)
        self.colon = vocab_map.get(":", 4)
        self.digits = {vocab_map[str(d)] for d in range(10) if str(d) in vocab_map}

    def process_logits(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.FloatTensor:
        masked_logits = logits.clone()
        last_token = input_ids[0, -1].item() if input_ids.shape[1] > 0 else None
        allowed_tokens: Set[int] = set()

        if last_token is None or last_token == self.bracket_close:
            allowed_tokens.add(self.bracket_open)
        elif last_token == self.bracket_open:
            allowed_tokens.update(self.digits)
        elif last_token in self.digits:
            allowed_tokens.update(self.digits)
            allowed_tokens.add(self.comma)
            allowed_tokens.add(self.colon)
            allowed_tokens.add(self.bracket_close)
        elif last_token in (self.comma, self.colon):
            allowed_tokens.update(self.digits)
        else:
            allowed_tokens.update(self.digits)

        mask = torch.full_like(masked_logits, fill_value=float("-inf"))
        for tok_id in allowed_tokens:
            mask[0, tok_id] = 0.0

        return masked_logits + mask
