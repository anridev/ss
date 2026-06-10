#! /usr/bin/env python3
"""Generate diverse synthetic extortion / sextortion SMS examples.

Extortion messages differ from typical promotional/phishing spam: they make a
compromise claim ("I hacked you / recorded you"), state what the sender
allegedly holds, threaten to leak it, and demand payment or contact. This
generator randomizes the claim, leverage, threat, demand, amounts, deadlines,
and contact handles (emails / crypto wallets / Telegram) so the model learns the
*pattern* rather than memorizing one fixed string.

Usage:
    ./gen_extortion.py                  # 150 examples -> generated_extortion.jsonl
    ./gen_extortion.py --n 250 --out foo.jsonl --seed 7
"""

import argparse
import json
import random
import string

OPENERS = [
    "I have some bad news for you.",
    "This is your final warning.",
    "I've been watching you for a while.",
    "Your device has been compromised.",
    "I gained full access to your phone.",
    "Your account has been hacked.",
    "Pay attention, this is not a joke.",
    "I am a hacker and I have access to your data.",
    "Do not ignore this message.",
    "I installed spyware on your device weeks ago.",
    "Hello. I know one of your passwords.",
    "Unfortunately for you, I have full control of your accounts.",
]

CLAIMS = [
    "I have copied all your contacts and private messages",
    "I have recorded everything through your camera",
    "I have full access to your photos and chat history",
    "I downloaded all the data you ever sent to {brand}",
    "I have your browsing history and saved passwords",
    "I have screenshots of your private conversations",
    "every file on your device is now in my possession",
    "I obtained your banking details and personal documents",
    "I have a video of you that you would not want anyone to see",
]

THREATS = [
    "I will send everything to all your contacts",
    "I will publish your data publicly",
    "I will leak it to your family and coworkers",
    "I will share it on social media",
    "I will expose you to everyone you know",
    "your reputation will be destroyed",
    "I will release the recording unless you act",
]

DEMAND_TEMPLATES = [
    "Pay {amount} in Bitcoin to {wallet} within {deadline}.",
    "Transfer {amount} worth of crypto to {wallet} before {deadline}.",
    "Send {amount} to the wallet {wallet} or everything goes public.",
    "Contact me at {email} within {deadline} if you want to keep this private.",
    "Reach out to {email} now to negotiate and avoid the leak.",
    "Message me on Telegram {tg} within {deadline}.",
    "The only way to stop this is to pay {amount} to {wallet}.",
]

BRANDS = ["SNB", "Al Rajhi", "your bank", "STC", "Absher", "your email provider", "Apple", "Google"]
AMOUNTS = ["$500", "$750", "$1000", "$1500", "$2000", "1500 SAR", "3000 SAR", "5000 SAR",
           "0.01 BTC", "0.02 BTC", "0.05 BTC", "300 USDT", "500 USDT"]
DEADLINES = ["24 hours", "48 hours", "72 hours", "two days", "by tomorrow", "the next 12 hours"]
EMAIL_DOMAINS = ["proton.me", "protonmail.com", "tutanota.com", "gmail.com", "outlook.com", "mail.com"]


def rand_token(rng, lo, hi, charset):
    return "".join(rng.choices(charset, k=rng.randint(lo, hi)))


def make_email(rng):
    return f"{rand_token(rng, 6, 10, string.ascii_lowercase + string.digits)}@{rng.choice(EMAIL_DOMAINS)}"


def make_wallet(rng):
    prefix = rng.choice(["1", "3", "bc1", "0x"])
    return prefix + rand_token(rng, 24, 30, string.ascii_letters + string.digits)


def make_telegram(rng):
    return "@" + rand_token(rng, 6, 11, string.ascii_lowercase + "_")


def build(rng):
    parts = []
    parts.append(rng.choice(OPENERS))
    claim = rng.choice(CLAIMS).format(brand=rng.choice(BRANDS))
    parts.append(claim + ".")
    if rng.random() < 0.85:
        parts.append(rng.choice(THREATS) + ".")
    demand = rng.choice(DEMAND_TEMPLATES).format(
        amount=rng.choice(AMOUNTS),
        wallet=make_wallet(rng),
        email=make_email(rng),
        tg=make_telegram(rng),
        deadline=rng.choice(DEADLINES),
    )
    parts.append(demand)
    if rng.random() < 0.4:
        parts.append(rng.choice([
            "Do not reply to this number.",
            "The clock is ticking.",
            "Do not contact the police.",
            "This is your only chance.",
        ]))
    return " ".join(parts)


def generate(n, seed):
    rng = random.Random(seed)
    out, seen = [], set()
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        msg = build(rng)
        if msg in seen:
            continue
        seen.add(msg)
        out.append({"text": msg, "label": 1, "category": "extortion"})
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic extortion/sextortion SMS examples")
    parser.add_argument("--n", type=int, default=150, help="Number of examples (default: 150)")
    parser.add_argument("--out", default="generated_extortion.jsonl", help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    rows = generate(args.n, args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} unique extortion examples to {args.out}")


if __name__ == "__main__":
    main()
