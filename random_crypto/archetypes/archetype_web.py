import random
import json
import base64
import time
import hashlib
from Crypto.Cipher import AES
from crypto_helpers import generate_flag, random_string
import string


def generate_jwt_none_algorithm_challenge():
    flag = generate_flag()
    
    header = json.dumps({"alg": "none", "typ": "JWT"}).encode()
    payload = json.dumps({"msg": flag}).encode()
    
    # Base64 URL encoding without padding
    def base64url_encode(data):
        return base64.urlsafe_b64encode(data).decode().rstrip("=")

    jwt_token = f"{base64url_encode(header)}.{base64url_encode(payload)}."

    return {
        "flag": flag,
        "ciphertext": jwt_token,
        "cipher": "JWT",
        "hint": "JWT tokens with 'alg: none'.",
        "difficulty": "easy",
    }


def generate_weak_cookie_encryption_challenge():
    flag = generate_flag()
    short_key = b"AA"  # 2 bytes only!
    
    # Pad the key to 16 bytes for AES-128
    key_16 = short_key.ljust(16, b"\x00")

    cipher = AES.new(key_16, AES.MODE_ECB)
    block_size = 16
    padded_cookie = flag.encode("utf-8")
    padding_len = block_size - (len(padded_cookie) % block_size)
    padded_cookie += bytes([padding_len]) * padding_len

    encrypted_cookie = cipher.encrypt(padded_cookie)

    return {
        "flag": flag,
        "ciphertext": encrypted_cookie.hex(),
        "cipher": "AES-ECB",
        "vulnerability": "small key padded with null bytes",
        "hint": "ECB mode with a short key makes brute-forcing possible. Consider the small key entropy.",
        "necessary_info": "An encryption key that is only 2 bytes long?.",
        "extra_info": "Padding follows PKCS#7. The key is padded to 16 bytes with null bytes.",
        "difficulty": "easy",
    }


def generate_broken_key_exchange_challenge():
    flag = generate_flag()

    small_prime = 101  
    generator = 2

    secret_a = random.randint(2, 99) 
    secret_b = random.randint(2, 99)

    public_a = pow(generator, secret_a, small_prime)
    public_b = pow(generator, secret_b, small_prime)
    shared_key = pow(public_b, secret_a, small_prime)

    # Derive a simple encryption key from the shared key
    encryption_key = hashlib.sha256(str(shared_key).encode()).digest()[:16]

    # XOR encryption (simple symmetric encryption)
    encrypted_flag = bytes(a ^ b for a, b in zip(flag.encode(), encryption_key))

    return {
        "flag": flag,
        "ciphertext": encrypted_flag.hex(),
        "cipher": "Diffie-Hellman",
        "hint": "The prime used in this key exchange is small. You can recover the private key by brute-force.",
        "necessary_info": (
            f"The key exchange uses a prime modulus of {small_prime} and generator {generator}.\n"
            f"Alice's public key is {public_a} and Bob's public key is {public_b}.\n"
            "You received a tip that the private values are at most two digits.\n"
            "You got access to some of the code:\n"
            "encryption_key = hashlib.sha256(str(shared_key).encode()).digest()[:16]\n"
            "encrypted_flag = bytes(a ^ b for a, b in zip(flag.encode(), encryption_key))"
        ),
        "extra_info": f"secret_a={secret_a}, secret_b={secret_b}",
        "difficulty": "medium"
    }


def generate_insecure_session_token_challenge():
    flag = generate_flag()
    username = "user_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    timestamp = str(int(time.time()))

    # Construct vulnerable token containing sensitive information
    raw_token = f"{username}:{flag}:{timestamp}"
    session_token = raw_token.encode("utf-8").hex()  # Simple hex encoding

    return {
        "flag": flag,
        "ciphertext": session_token,
        "cipher": "Insecure Session Token",
        "hint": "Hex decoding is reversible. Session tokens often contain structured data.",
        "extra_info": f"Token generation code: '{raw_token}'.encode('utf-8').hex()",
        "difficulty": "easy",
    }


def generate_web_crypto_challenge(variation):

    if variation == "jwt_none_algorithm":
        return generate_jwt_none_algorithm_challenge()
    elif variation == "weak_cookie_encryption":
        return generate_weak_cookie_encryption_challenge()
    elif variation == "broken_key_exchange":
        return generate_broken_key_exchange_challenge()
    elif variation == "insecure_session_token":
        return generate_insecure_session_token_challenge()
    else:
        # Fallback for unrecognized variation
        plaintext = generate_flag()
        return {
            "archetype": "web_crypto",
            "vulnerability": variation,
            "plaintext": plaintext,
            "info": "Web crypto challenge not recognized or not yet implemented."
        }

if __name__ == "__main__":
    for var in [
        "jwt_none_algorithm",
        "weak_cookie_encryption",
        "broken_key_exchange",
        "insecure_session_token"
    ]:
        challenge = generate_web_crypto_challenge(var)
        print(f"--- {var.upper()} ---")
        for k, v in challenge.items():
            print(f"{k}: {v}")
        print()
