import random
import string
import hashlib
import os
from crypto_helpers import generate_flag

def generate_md5_reverse_challenge():
    word_list_path = os.path.join(os.path.dirname(__file__), "md5_reverse_wordlist.txt")
    with open(word_list_path, "r", encoding="utf-8") as file:
        passwords = file.readlines()
    random_pass = random.choice(passwords).strip()
    hash_value = hashlib.md5(random_pass.encode('utf-8')).hexdigest()

    flag = "flag{" + random_pass + "}"
    
    return {
        "flag": flag,
        "ciphertext": hash_value,
        "cipher": "MD5",
        "vulnerability": "MD5 reversing",
        "hint": "The input string is short, so a brute force or dictionary attack might succeed.",
        "necessary_info": "The flag format is flag{...}",
        "difficulty": "hard"
    }


def generate_poor_random_salt_challenge():
    salt = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(3))  # 3-char salt
    hashed_val = hashlib.sha256((salt + "apple").encode('utf-8')).hexdigest()

    flag = "flag{" + salt +"}"
    return {
        "flag": flag,
        "ciphertext": hashed_val,
        "cipher": "SHA-256",
        "vulnerability": "poor random salt",
        "hint": "You can brute-force the solution.",
        "necessary_info": (
            "The salt is the flag with the format flag{salt}.\n"
            "The code that generated the flag:\n"
            "salt = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(3))"
            """hashlib.sha256((salt + "apple").encode('utf-8')).hexdigest()"""
        ),
        "difficulty": "easy",
    }


def generate_iterated_hash_challenge():
    secret = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(4))

    iterated_hash = secret
    for _ in range(5):
        iterated_hash = hashlib.sha256(iterated_hash.encode('utf-8')).hexdigest()

    flag = "flag{" + secret + "}"

    return {
        "flag": flag,
        "ciphertext": iterated_hash,
        "cipher": "SHA-256",
        "hint": "Brute force should work.",
        "necessary_info": (
            "The secret is only 4 characters (a-z, 0-9).\n"
            "The flag format is flag{secret}\n"
            "The hashing process was:\n"
            "hash = secret\n"
            "for _ in range(5):\n"
            "    hash = hashlib.sha256(hash.encode('utf-8')).hexdigest()"
        ),
        "difficulty": "medium"
    }


def generate_hash_challenge(variation):
    if variation == "md5_reverse":
        return generate_md5_reverse_challenge()
    elif variation == "poor_random_salt":
        return generate_poor_random_salt_challenge()
    elif variation == "iterated_hash_challenge":
        return generate_iterated_hash_challenge()
    else:
        # Fallback for unrecognized variation
        return {
            "flag": generate_flag(),
            "challenge_type": "hash",
            "vulnerability": variation,
            "info": f"Hash challenge '{variation}' not implemented."
        }



if __name__ == "__main__":
    for var in ["length_extension", "collision_attack", "poor_random_salt"]:
        print(f"--- {var.upper()} ---")
        challenge = generate_hash_challenge(var)
        for k, v in challenge.items():
            print(f"{k}: {v}")
        print()
