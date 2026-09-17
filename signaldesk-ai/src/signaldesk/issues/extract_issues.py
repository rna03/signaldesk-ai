"""Extract short issue statements with a local, pretrained FLAN-T5 model."""

import argparse
from collections import Counter, defaultdict
from functools import cached_property
import json
from pathlib import Path
from random import Random
import re
from time import perf_counter

from signaldesk.clustering.discover_issues_semantic import ARTIFACT_DIR, load_conversations
from signaldesk.data.inspect_dataset import DATASET_NAME


MODEL_NAME = "google/flan-t5-small"
PROMPT_VERSION = 6
MAX_INPUT_TOKENS = 128  # Deliberate opening window; below FLAN-T5-small n_positions=512.
MAX_NEW_TOKENS = 32
BATCH_SIZE = 8
SAMPLE_SEED = 42
ARTIFACT_PATH = ARTIFACT_DIR / "extracted_issues.json"
GENERIC_OUTPUTS = {
    "no i'm not sure", "no i am not sure", "i'm not sure", "i am not sure",
    "i'm sorry", "i am sorry", "i'm sorry i'm sorry", "i don't know",
    "i don't know what to do", "i'm calling", "no i'm calling",
    "i need to know", "i need to know more about that", "i need to know the details",
    "if you have any questions", "a few more", "i will get issue", "i've got a",
}
NUMBER_WORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
REQUEST_CUE = re.compile(
    r"\b(?:i['’]d\s+like\s+to|calling\s+(?:because|about|to)|looking\s+for|"
    r"would\s+like\s+to|"
    r"(?:i|we)\s+(?:want|wanted|need|needed|have)\s+to|wondering\s+if|"
    r"talk\s+about|can't|cannot|couldn't|doesn't|didn't|haven't|"
    r"problem\s+with|issue\s+with|supposed\s+to)\b",
    flags=re.IGNORECASE,
)


def build_prompt(customer_text: str) -> str:
    """Keep the instruction identical for every conversation."""
    return (
        "Extract the customer's main request or complaint as one short phrase from the text. "
        "Do not answer, describe the agent, or invent details.\n"
        f"Customer: {customer_text}\nIssue:"
    )


def select_focus_text(customer_text: str) -> tuple[str, bool]:
    """Start near the first explicit request cue; keep the source transcript intact."""
    match = REQUEST_CUE.search(customer_text)
    if match is None:
        return customer_text, False
    words = list(re.finditer(r"\S+", customer_text))
    cue_index = next(index for index, word in enumerate(words) if word.end() > match.start())
    start = words[max(0, cue_index - 2)].start()
    prior_sentence_end = list(re.finditer(r"[.!?]\s+", customer_text[start:match.start()]))
    if prior_sentence_end:
        start += prior_sentence_end[-1].end()
    end_index = min(len(words), cue_index + 45)
    end = words[end_index - 1].end()
    return customer_text[start:end], start > 0 or end < len(customer_text)


def prepare_prompt(tokenizer, customer_text: str, max_input_tokens: int = MAX_INPUT_TOKENS):
    """Measure token length and, if necessary, keep a deterministic text prefix."""
    if not isinstance(customer_text, str) or not customer_text.strip():
        raise ValueError("customer_text must be nonempty")
    fixed_tokens = len(tokenizer.encode(build_prompt(""), add_special_tokens=True, verbose=False))
    available = max_input_tokens - fixed_tokens - 8  # Space for token boundary changes.
    if available <= 0:
        raise ValueError("Token limit is too small for the extraction prompt")
    focused_text, shifted = select_focus_text(customer_text)
    # tokenize() has no model-length warning; we inspect the length before generate().
    customer_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(focused_text))
    truncated = shifted or len(customer_ids) > available
    if not truncated:
        prompt = build_prompt(focused_text)
        if len(tokenizer.encode(prompt, add_special_tokens=True, verbose=False)) <= max_input_tokens:
            return prompt, False
        truncated = True

    # Decode only the selected customer tokens; the original transcript stays intact.
    while available > 0:
        shortened = tokenizer.decode(
            customer_ids[:available], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        prompt = build_prompt(shortened)
        if len(tokenizer.encode(prompt, add_special_tokens=True, verbose=False)) <= max_input_tokens:
            return prompt, truncated
        available -= 8
    raise ValueError("Could not fit customer text and prompt within the model limit")


def normalize_issue(generated_text: str) -> str:
    """Remove formatting only; do not stem or drop meaningful words."""
    if not isinstance(generated_text, str):
        return ""
    value = re.sub(r"\s+", " ", generated_text).strip()
    value = re.sub(r"^issue(?: statement)?\s*:\s*", "", value, flags=re.IGNORECASE).strip()
    if value.casefold() in {"", "n/a", "none", "unknown", "no issue"} or re.fullmatch(
        r"[\W_]*(?:um|uh|ah|er|hmm)[\W_]*", value, flags=re.IGNORECASE
    ):
        return ""
    canonical = re.sub(r"[^a-z0-9']+", " ", value.casefold()).strip()
    if canonical in GENERIC_OUTPUTS or "anything else i can help you with" in canonical:
        return ""
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", value)
    if len(words) < 3 or len({word.casefold() for word in words}) == 1:
        return ""
    if not any(len(word) > 1 and word.casefold() not in NUMBER_WORDS for word in words):
        return ""
    if re.match(r"^[A-Z]{2,5}\s*[- ]*\d", value):
        return ""  # Looks like a booking/account code, not an issue statement.
    return value


def fallback_statement(customer_text: str) -> str:
    """Use a visible transcript excerpt when generation fails, never a fake model claim."""
    focused_text, _ = select_focus_text(customer_text)
    excerpt = focused_text.strip()
    sentence_end = re.search(r"[.!?](?:\s|$)", excerpt)
    if sentence_end is not None and sentence_end.end() >= 25:
        excerpt = excerpt[:sentence_end.start() + 1]
    return excerpt[:180].strip()


class IssueExtractor:
    """One cached tokenizer/model pair; no training or fine-tuning."""

    def __init__(self, tokenizer=None, model=None):
        if tokenizer is not None:
            self.__dict__["tokenizer"] = tokenizer
        if model is not None:
            self.__dict__["model"] = model

    @cached_property
    def tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(MODEL_NAME)

    @cached_property
    def model(self):
        from transformers import AutoModelForSeq2SeqLM

        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        model.eval()
        return model

    def extract_batch(self, customer_texts: list[str]) -> list[dict]:
        if not customer_texts:
            return []
        input_limit = min(MAX_INPUT_TOKENS, int(self.model.config.n_positions))
        prepared = [prepare_prompt(self.tokenizer, text, input_limit) for text in customer_texts]
        prompts = [prompt for prompt, _ in prepared]
        tokenized = self.tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
        if tokenized["input_ids"].shape[1] > input_limit:
            raise ValueError("Prepared prompt unexpectedly exceeds model token limit")
        generated_ids = self.model.generate(
            **tokenized, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1
        )
        outputs = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if len(outputs) != len(customer_texts):
            raise RuntimeError("Model returned a different number of issue statements")
        result = []
        for text, output, (_, truncated) in zip(customer_texts, outputs, prepared):
            statement = normalize_issue(output)
            source = "flan_t5" if statement else "fallback"
            result.append({
                "issue_statement": statement or fallback_statement(text),
                "issue_source": source,
                "input_truncated": truncated,
            })
        return result


def select_domain_sample(conversations: list[dict], count: int = 10, seed: int = SAMPLE_SEED):
    """Select one reproducible real conversation from each of distinct domains."""
    by_domain = defaultdict(list)
    for conversation in conversations:
        by_domain[conversation["domain"]].append(conversation)
    if len(by_domain) < count:
        raise ValueError(f"Need {count} distinct domains; found {len(by_domain)}")
    rng = Random(seed)
    return [
        rng.choice(sorted(by_domain[domain], key=lambda row: row["conversation_id"]))
        for domain in sorted(by_domain)[:count]
    ]


def extract_conversations(conversations: list[dict], extractor: IssueExtractor, batch_size=BATCH_SIZE):
    """Preserve source order and original customer text in each derived record."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    records = []
    for start in range(0, len(conversations), batch_size):
        group = conversations[start:start + batch_size]
        extracted = extractor.extract_batch([row["customer_text"] for row in group])
        if len(extracted) != len(group):
            raise RuntimeError("Extractor returned a different number of records")
        for conversation, issue in zip(group, extracted):
            records.append({
                "conversation_id": conversation["conversation_id"],
                "domain": conversation["domain"],
                "customer_text": conversation["customer_text"],
                **issue,
            })
        if len(conversations) > 10 and (len(records) % 80 == 0 or len(records) == len(conversations)):
            print(f"Extracted {len(records)}/{len(conversations)} conversations", flush=True)
    return records


def make_artifact(records: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "dataset": DATASET_NAME,
        "model": MODEL_NAME,
        "prompt_version": PROMPT_VERSION,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "generation": "greedy_do_sample_false",
        "conversation_count": len(records),
        "records": records,
    }


def validate_artifact(artifact: dict) -> list[dict]:
    """Reject incomplete/stale generated data before clustering it."""
    if not isinstance(artifact, dict) or any(
        artifact.get(key) != value for key, value in {
            "schema_version": 1,
            "dataset": DATASET_NAME,
            "model": MODEL_NAME,
            "prompt_version": PROMPT_VERSION,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "generation": "greedy_do_sample_false",
        }.items()
    ):
        raise ValueError("Extracted issue artifact metadata does not match this experiment")
    records = artifact.get("records")
    if not isinstance(records, list) or artifact.get("conversation_count") != len(records):
        raise ValueError("Extracted issue artifact has an invalid record count")
    seen = set()
    for row in records:
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ("conversation_id", "domain", "customer_text", "issue_statement")
        ) or row.get("issue_source") not in {"flan_t5", "fallback"} or not isinstance(
            row.get("input_truncated"), bool
        ):
            raise ValueError("Extracted issue artifact contains an invalid record")
        if row["conversation_id"] in seen:
            raise ValueError("Extracted issue artifact contains duplicate conversation IDs")
        seen.add(row["conversation_id"])
    return records


def write_artifact(records: list[dict], path: Path = ARTIFACT_PATH):
    artifact = make_artifact(records)
    validate_artifact(artifact)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")


def load_artifact(path: Path = ARTIFACT_PATH) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Extracted issues are missing. Run python -m signaldesk.issues.extract_issues --full"
        )
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Extracted issue artifact cannot be read as JSON") from exc
    return validate_artifact(artifact)


def main():
    parser = argparse.ArgumentParser(description="Local FLAN-T5 customer issue extraction")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sample", action="store_true", help="Inspect one conversation from each of 10 domains")
    mode.add_argument("--full", action="store_true", help="Extract all conversations and write artifact")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    conversations = load_conversations()
    chosen = select_domain_sample(conversations) if args.sample else conversations
    print(f"Model: {MODEL_NAME}; pretrained inference only; CPU/local baseline", flush=True)
    print(f"Mode: {'10-domain sample' if args.sample else 'full dataset'}; conversations: {len(chosen)}", flush=True)
    start = perf_counter()
    records = extract_conversations(chosen, IssueExtractor(), args.batch_size)
    runtime = perf_counter() - start

    counts = Counter(row["issue_source"] for row in records)
    print(f"Extraction runtime including model load: {runtime:.2f} seconds")
    print(f"flan_t5: {counts['flan_t5']}; fallback: {counts['fallback']}")
    print(f"Empty issue count: {sum(not row['issue_statement'].strip() for row in records)}")
    print(f"Input truncated count: {sum(row['input_truncated'] for row in records)}")
    if args.sample:
        for row in records:
            preview = row["customer_text"].replace("\n", " ")[:320]
            print(f"\nDOMAIN: {row['domain']}")
            print(f"CUSTOMER TEXT (display preview): {preview}{'...' if len(row['customer_text']) > 320 else ''}")
            print(f"EXTRACTED ISSUE: {row['issue_statement']}")
            print(f"SOURCE: {row['issue_source']}; input_truncated: {row['input_truncated']}")
    else:
        write_artifact(records)
        print(f"Artifact: {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
