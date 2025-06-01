import os
import csv
import re
import argparse
import sys
sys.path.append("archetypes")
import dotenv
dotenv.load_dotenv()

from openai import OpenAI

from archetypes.archetype_classical import generate_classical_cipher_challenge
from archetypes.archetype_aes import generate_aes_challenge
from archetypes.archetype_rsa import generate_rsa_challenge
from archetypes.archetype_ecc import generate_ecc_challenge
from archetypes.archetype_hash import generate_hash_challenge
from archetypes.archetype_prng import generate_prng_challenge
from archetypes.archetype_web import generate_web_crypto_challenge
from archetypes.archetype_signature import generate_signature_schemes_challenge


client = OpenAI()

# Weighted distribution of top-level archetypes.
ARCHETYPES = [
    "classical_cipher",
    "rsa",
    "aes",
    "ecc",
    "hash",
    "prng",
    "web_crypto",
    "signature_schemes",
]

# SUBTYPES that have been implemented in crypto_helpers.py.
SUBTYPES = {
    "classical_cipher": [
        "caesar",
        "vigenere",
        "playfair",
        "hill",
        "rail_fence",
        "substitution",
        "transposition",
        "autokey",
        "base64_layered",
        "morse_code",
        "fibonacci_encoding",
        "XOR",
        "Base64",
        "Base64_layered",
        "Base85",
        "Base85_layered",
        "substitution_direct",
        "atbash",
        "hex",
        "ascii_shift",
        "split_flag",
        "reversed_flag",
        "chunked_flag",
    ],
    "rsa": [
        "small_primes",
        "repeated_prime_usage",
        "partial_key_exposure",
        "common_factors",
        "shared_prime",
        "blum_integers"
    ],
    "aes": [
        "aes_gcm",
        "aes_ccm",
        "aes_xts",
        "aes_cfb",
    ],
    "ecc": [
        "small_order_curves",
        "faulty_curve_parameters",
        "reused_nonce_ecdsa"
    ],
    "hash": [
        "md5_reverse",
        "poor_random_salt",
        "iterated_hash_challenge"
    ],
    "prng": [
        "predictable_seed",
        "time_based_seed",
        "low_entropy_generator",
        "lfsr_weakness",
        "congruential_generator_flaw"
    ],
    "web_crypto": [
        "jwt_none_algorithm",
        "weak_cookie_encryption",
        "broken_key_exchange",
        "insecure_session_token"
    ],
    "signature_schemes": [
        "ecdsa_nonce_reuse",
        "rsa_sign_with_low_public_exponent",
    ],
}


def generate_story_using_llm(challenge_data):
    #return "TESTING:" + str(challenge_data)

    if "vulnerability" in challenge_data:
        vulnerability_text = f" and the {challenge_data['vulnerability']} vulnerability"
    else:
        vulnerability_text = ""

    user_prompt = f"""
        Write a *very short* (≤ 30 words) back-story for a crypto
        challenge that uses the {challenge_data['cipher']} cipher{vulnerability_text}.

        Keep it *very brief* without unnecesary information.

        Return *only* the text between these XML tags:
        <challenge>…</challenge>

        Insert the placeholder <CIPHER> **verbatim** where the ciphertext will go.
        Do *not* write anything outside the tags.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a creative puzzle-writing assistant."},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=512
        )
        response = response.choices[0].message.content.strip()
        story_text = re.findall(r"<challenge>(.*?)<\/challenge>", response, re.DOTALL)[0]
        story_text = story_text.replace("<CIPHER>", challenge_data["ciphertext"])
        return story_text

    except Exception as e:
        print(e)
        return (f"## LLM Story Generation Failed\n"
                f"Error: {str(e)}\n"
                f"Please see challenge data below:\n{challenge_data}")


def generate_every_possible_challenge(output_folder, variants=1):
    challanges = []
    for i in range(variants):
        for archetype in SUBTYPES:
            for subtype in SUBTYPES[archetype]:
                if archetype == "classical_cipher":
                    challenge_data = generate_classical_cipher_challenge(subtype)
                elif archetype == "rsa":
                    challenge_data = generate_rsa_challenge(subtype)
                elif archetype == "aes":
                    challenge_data = generate_aes_challenge(subtype)
                elif archetype == "ecc":
                    challenge_data = generate_ecc_challenge(subtype)
                elif archetype == "hash":
                    challenge_data = generate_hash_challenge(subtype)
                elif archetype == "prng":
                    challenge_data = generate_prng_challenge(subtype)
                elif archetype == "web_crypto":
                    challenge_data = generate_web_crypto_challenge(subtype)
                elif archetype == "signature_schemes":
                    challenge_data = generate_signature_schemes_challenge(subtype)
                else:
                    challenge_data = {
                        "info": "Placeholder - no implementation yet for this archetype."
                    }

                challenge_data["subtype"] = subtype
                challenge_data["archetype"] = archetype
                
                story_text = generate_story_using_llm(challenge_data)
                if "necessary_info" in challenge_data:
                    challenge_data["question"] = story_text + "\n\n" + challenge_data["necessary_info"]
                else:
                    challenge_data["question"] = story_text
                challanges.append(challenge_data)
                save_challenge_to_file(challenge_data=challenge_data, challenge_index=len(challanges), output_folder=output_folder)
    return challanges


def save_challenge_to_file(challenge_data, challenge_index, output_folder):
    if not os.path.exists(os.path.join(output_folder, "challenges")):
        os.makedirs(os.path.join(output_folder, "challenges"))

    filename = f"challenge_{challenge_index}_{challenge_data['archetype']}_{challenge_data['subtype']}.txt"
    filepath = os.path.join(os.path.join(output_folder, "challenges"), filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(challenge_data["question"])
        f.write("\n\n---\n\n")
        f.write("## Challenge Technical Details\n")
        for key, val in challenge_data.items():
            if key == "question":
                continue
            f.write(f"{key}: {val}\n")

    print(f"[+] Challenge saved to: {filepath}")


def save_challenges_to_csv(challenges, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Collect all possible keys (columns) from every challenge_data
    all_keys = set()
    for challenge in challenges:
        for k in challenge.keys():
            all_keys.add(k)
    fieldnames = list(all_keys)

    csv_path = os.path.join(output_folder, "all_challenges.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for challenge in challenges:
            writer.writerow(challenge)
    
    print(f"[+] All challenges CSV saved to: {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an LLM on a procedural dataset with Pass@N and Majority@N.",
    )

    # Dataset
    parser.add_argument(
        "--variants",
        type=int,
        default=1,
        help="Number of variants per challenge subtipe (total challenges generated = variants * 50)",
    )

    parser.add_argument(
        "--output_folder", 
        default="generated_challenges_llm",
        type=str,
        help="Folder where results will be saved."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    challenges = generate_every_possible_challenge(variants=args.variants, output_folder=args.output_folder)

    print(f"Finished generating {len(challenges)} challenges.")

    save_challenges_to_csv(challenges, output_folder=args.output_folder)


if __name__ == "__main__":
    main()
