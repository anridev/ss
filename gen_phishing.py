#! /usr/bin/env python3
"""Generate diverse synthetic phishing (smishing) SMS examples.

Produces varied messages across several phishing sub-types (prize/giveaway,
bank alerts, delivery fees, government services, subscriptions, job/investment
scams, account lockouts, refunds, OTP theft), randomizing brands, amounts,
URLs, and phrasing so the model learns generalizable signals rather than
memorizing a handful of fixed strings.

Usage:
    ./gen_phishing.py                       # 300 examples -> generated_phishing.jsonl
    ./gen_phishing.py --n 500 --out foo.jsonl --seed 7
"""

import argparse
import json
import random
import string

BANKS = [
    "Al Rajhi Bank", "Alinma Bank", "Bank Albilad", "Riyad Bank",
    "Saudi National Bank", "Arab National Bank", "Saudi Awwal Bank",
    "Bank AlJazira", "Gulf International Bank", "Banque Saudi Fransi",
]
TELECOMS = ["STC", "Mobily", "Zain", "Lebara", "Virgin Mobile", "Salam Mobile"]
GOVT = [
    "Absher", "Tawakkalna", "Najm", "Muqeem", "ZATCA", "Saudi Post",
    "GOSI", "Ministry of Interior", "Ministry of Health",
]
DELIVERY = ["FedEx", "DHL", "Aramex", "SMSA Express", "Naqel Express", "UPS", "Saudi Post"]
BRANDS = ["Jarir", "noon", "Amazon", "Apple", "Netflix", "Amazon Prime",
          "Almarai", "Extra", "AliExpress", "Shein"]

SHORTENERS = ["bit.ly", "tinyurl.com", "is.gd", "t.ly", "rb.gy", "cutt.ly", "shorturl.at", "ow.ly"]
FAKE_DOMAINS = [
    "absher-gov.net", "ksa-prize.vip", "zatca-refund.co", "saudi-post.work",
    "sa-verify.top", "secure-login.info", "pay-fees.click", "claim-now.xyz",
    "stc-rewards.online", "win-now.live", "verify-id.cyou", "my-account.fit",
    "track-pkg.site", "gov-sa.help", "account-update.live", "rajhi-secure.net",
]

SMALL_FEES = [15, 19, 25, 29, 35, 45, 50, 99]
PRIZE_AMOUNTS = [500, 1500, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000]
HOURS = [2, 12, 24, 48, 72]


def rand_token(rng, lo=6, hi=9):
    n = rng.randint(lo, hi)
    return "".join(rng.choices(string.ascii_letters + string.digits, k=n))


def make_url(rng):
    style = rng.random()
    token = rand_token(rng)
    if style < 0.5:
        host = rng.choice(SHORTENERS)
    else:
        host = rng.choice(FAKE_DOMAINS)
    scheme = rng.choice(["", "http://", "https://"])
    return f"{scheme}{host}/{token}"


def sar(rng, pool):
    return f"{rng.choice(pool):,} SAR"


def code(rng):
    return f"{rng.randint(1000, 9999)}"


def ref(rng):
    return f"{rng.choice(['#', 'Ref #', 'PKG', 'SA'])}{rng.randint(1000000, 99999999)}"


# Each builder returns a single message string. They intentionally vary phrasing.
def b_prize(rng):
    src = rng.choice(BRANDS + TELECOMS + ["the national lottery", "the Eid draw"])
    templates = [
        f"Congratulations! Your number won {sar(rng, PRIZE_AMOUNTS)} in the {src} draw. Claim within {rng.choice(HOURS)} hours: {make_url(rng)}",
        f"{src} giveaway! You have been selected to receive {sar(rng, PRIZE_AMOUNTS)}. Reply with your IBAN here: {make_url(rng)}",
        f"WINNER! {src} has awarded your number a prize. Verify immediately to claim: {make_url(rng)}",
        f"Your mobile number was drawn to win an {rng.choice(['iPhone 17', 'Galaxy S25', 'luxury car', 'PlayStation 5'])}. Pay a {rng.choice(SMALL_FEES)} SAR delivery fee: {make_url(rng)}",
        f"FINAL NOTICE: Your {sar(rng, PRIZE_AMOUNTS)} cash prize expires in {rng.choice(HOURS)} hours. Claim here: {make_url(rng)}",
    ]
    return rng.choice(templates)


def b_bank(rng):
    bank = rng.choice(BANKS)
    templates = [
        f"{bank}: Your account has been temporarily suspended. Verify your identity now to avoid permanent closure: {make_url(rng)}",
        f"{bank} Alert: Login from a new device detected. If this wasn't you, secure your account: {make_url(rng)}",
        f"{bank}: A payment of {sar(rng, PRIZE_AMOUNTS)} was attempted from your account. To cancel, confirm your details: {make_url(rng)}",
        f"Dear customer, your {bank} card will be blocked within {rng.choice(HOURS)} hours. Update your information here: {make_url(rng)}",
        f"{bank}: Unusual activity detected. Reactivate your account within {rng.choice(HOURS)} hours: {make_url(rng)}",
        f"{bank}: You have {rng.randint(5, 90)},{rng.randint(100,999)} reward points worth {sar(rng, PRIZE_AMOUNTS)} expiring soon. Redeem now: {make_url(rng)}",
    ]
    return rng.choice(templates)


def b_delivery(rng):
    d = rng.choice(DELIVERY)
    templates = [
        f"{d}: Your parcel is on hold due to unpaid customs fees of {rng.choice(SMALL_FEES)} SAR. Pay now: {make_url(rng)}",
        f"{d}: Delivery attempted but your address is incomplete. Update it here: {make_url(rng)} ({ref(rng)})",
        f"Your package {ref(rng)} could not be delivered. Reschedule by tomorrow: {make_url(rng)}",
        f"{d} notice: A shipping fee of {rng.choice(SMALL_FEES)} SAR is required to release shipment {ref(rng)}: {make_url(rng)}",
        f"{d}: Your shipment {ref(rng)} is waiting. Confirm delivery details by tomorrow or it will be returned: {make_url(rng)}",
    ]
    return rng.choice(templates)


def b_govt(rng):
    g = rng.choice(GOVT)
    templates = [
        f"{g}: Your account requires re-verification. Failure to verify within {rng.choice(HOURS)} hours will suspend your services: {make_url(rng)}",
        f"{g}: A tax refund of {sar(rng, PRIZE_AMOUNTS)} is available. Submit your bank details to receive it: {make_url(rng)}",
        f"{g}: You have an unpaid violation of {sar(rng, PRIZE_AMOUNTS)}. Pay within {rng.choice(HOURS)} hours to avoid penalty: {make_url(rng)}",
        f"{g}: Update your information to keep your services active. Verify before 8 PM: {make_url(rng)}",
        f"{g}: Your residency (Iqama) expires by tomorrow. Renew now to avoid a fine: {make_url(rng)}",
        f"{g}: Your support payment of {sar(rng, PRIZE_AMOUNTS)} is pending. Confirm your IBAN here: {make_url(rng)}",
    ]
    return rng.choice(templates)


def b_subscription(rng):
    s = rng.choice(BRANDS)
    templates = [
        f"{s}: Your subscription payment failed. Update your card to avoid suspension: {make_url(rng)}",
        f"{s}: We could not renew your plan. Confirm your payment details immediately: {make_url(rng)}",
        f"{s}: Your account will be charged {sar(rng, PRIZE_AMOUNTS)} today. To cancel, verify here: {make_url(rng)}",
    ]
    return rng.choice(templates)


def b_job(rng):
    templates = [
        f"Work from home and earn {sar(rng, PRIZE_AMOUNTS)}/day. Limited spots. Apply now: {make_url(rng)}",
        f"Your CV was shortlisted for a {sar(rng, PRIZE_AMOUNTS)}/month remote role. Complete registration today: {make_url(rng)}",
        f"Investment opportunity: turn {sar(rng, PRIZE_AMOUNTS)} into profit in {rng.randint(3,14)} days. Start here: {make_url(rng)}",
        f"Earn daily profits with our trading platform. Deposit {sar(rng, PRIZE_AMOUNTS)} and start now: {make_url(rng)}",
    ]
    return rng.choice(templates)


def b_account(rng):
    s = rng.choice(BRANDS + ["your email", "your account"])
    templates = [
        f"Your {s} has been locked due to suspicious activity. Unlock it now: {make_url(rng)}",
        f"Your account will be deleted by tomorrow unless you confirm your identity: {make_url(rng)}",
        f"Security alert: your password was changed. If this wasn't you, reset it immediately: {make_url(rng)}",
        f"Your verification code is {code(rng)}. Do NOT share it. If you didn't request it, secure your account: {make_url(rng)}",
    ]
    return rng.choice(templates)


def b_donation(rng):
    templates = [
        f"Support families in need this Ramadan. Donate {sar(rng, PRIZE_AMOUNTS)} securely: {make_url(rng)}",
        f"Your donation of {sar(rng, PRIZE_AMOUNTS)} is one click away. Complete it here: {make_url(rng)}",
        f"Emergency relief fund: contribute {sar(rng, PRIZE_AMOUNTS)} now to help victims: {make_url(rng)}",
    ]
    return rng.choice(templates)


BUILDERS = [b_prize, b_bank, b_delivery, b_govt, b_subscription, b_job, b_account, b_donation]
PREFIXES = ["", "", "", "URGENT: ", "Important: ", "Notice: ", "Action required: "]


def generate(n, seed):
    rng = random.Random(seed)
    out, seen = [], set()
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        msg = rng.choice(PREFIXES) + rng.choice(BUILDERS)(rng)
        if msg in seen:
            continue
        seen.add(msg)
        out.append({"text": msg, "label": 1, "category": "phishing"})
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic phishing SMS examples")
    parser.add_argument("--n", type=int, default=300, help="Number of examples (default: 300)")
    parser.add_argument("--out", default="generated_phishing.jsonl", help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    rows = generate(args.n, args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} unique phishing examples to {args.out}")


if __name__ == "__main__":
    main()
