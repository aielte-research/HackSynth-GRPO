from crypto_helpers import generate_flag, pkcs7_pad
import os
import random

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad

from cryptography.hazmat.primitives.ciphers import Cipher as CryptoCipher
from cryptography.hazmat.primitives.ciphers import algorithms, modes
from cryptography.hazmat.backends import default_backend


def generate_aes_gcm_challenge():
    flag = generate_flag()
    second_text = "this is not a flag"  # A second flag to demonstrate nonce reuse vulnerability
    key = get_random_bytes(16)  # 128-bit key
    nonce = get_random_bytes(12)  # Weak nonce: the same for both encryptions

    # Encrypt the first flag
    cipher1 = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext1, tag1 = cipher1.encrypt_and_digest(flag.encode('utf-8'))

    # Encrypt the second flag with the same nonce (vulnerability)
    cipher2 = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext2, tag2 = cipher2.encrypt_and_digest(second_text.encode('utf-8'))

    return {
        "flag": flag,
        "ciphertext": f"ciphertext1:{ciphertext1.hex()}, ciphertext2: {ciphertext2.hex()}",
        "tag1": tag1.hex(),
        "tag2": tag2.hex(),
        "cipher": "AES-GCM",
        "vulnerability": "nonce reuse",
        "hint": "AES-GCM with nonce reuse is vulnerable. Exploit the relationship between the ciphertexts to recover the flag.",
        "necessary_info": "The flag format is flag{...}",
        "extra_info": f"The nonce used for both encryptions is: {nonce.hex()}",
        "difficulty": "hard"
    }


def generate_aes_ccm_challenge():
    flag = generate_flag()
    second_text = generate_flag()  # A second flag to demonstrate nonce reuse vulnerability
    key = get_random_bytes(16)  # 128-bit key
    nonce = get_random_bytes(11)  # Weak nonce: the same for both encryptions

    # Encrypt the first flag
    cipher1 = AES.new(key, AES.MODE_CCM, nonce=nonce)
    ciphertext1 = cipher1.encrypt(flag.encode('utf-8'))
    tag1 = cipher1.digest()

    # Encrypt the second flag with the same nonce (vulnerability)
    cipher2 = AES.new(key, AES.MODE_CCM, nonce=nonce)
    ciphertext2 = cipher2.encrypt(second_text.encode('utf-8'))
    tag2 = cipher2.digest()

    return {
        "flag": flag,
        "ciphertext": f"ciphertext1:{ciphertext1.hex()}, ciphertext2: {ciphertext2.hex()}",
        "tag1": tag1.hex(),
        "tag2": tag2.hex(),
        "cipher": "AES",
        "mode": "CCM",
        "hint": "AES-CCM with nonce reuse is vulnerable. Exploit the relationship between the ciphertexts to recover the flag.",
        "necessary_info": "The flag format is flag{...}",
        "extra_info": f"The nonce used for both encryptions is: {nonce.hex()}",
        "difficulty": "medium"
    }


def generate_aes_xts_challenge():
    flag = generate_flag()
    second_text = "this is not the flag"  # A second plaintext to demonstrate tweak reuse vulnerability
    full_key = get_random_bytes(32)  # 32-byte key for AES-128-XTS
    tweak = get_random_bytes(16)  # Reused tweak for both encryptions

    # Pad the texts if needed
    padded_flag1 = pad(flag.encode('utf-8'), 16)
    padded_flag2 = pad(second_text.encode('utf-8'), 16)

    # Encrypt the first flag
    cipher1 = CryptoCipher(
        algorithms.AES(full_key), 
        modes.XTS(tweak), 
        backend=default_backend()
    )
    encryptor1 = cipher1.encryptor()
    ciphertext1 = encryptor1.update(padded_flag1) + encryptor1.finalize()

    # Encrypt the second text with the same tweak (vulnerability)
    cipher2 = CryptoCipher(
        algorithms.AES(full_key), 
        modes.XTS(tweak), 
        backend=default_backend()
    )
    encryptor2 = cipher2.encryptor()
    ciphertext2 = encryptor2.update(padded_flag2) + encryptor2.finalize()

    return {
        "flag": flag,
        "ciphertext": f"ciphertext1:{ciphertext1.hex()}, ciphertext2: {ciphertext2.hex()}",
        "cipher": "AES-XTS",
        "hint": "AES-XTS with tweak reuse is vulnerable. Exploit the relationship between the ciphertexts to recover the flag.",
        "necessary_info": "The flag format is flag{...}",
        "extra_info": f"The tweak used for both encryptions is: {tweak.hex()}",
        "difficulty": "hard"
    }


def generate_aes_cfb_challenge():
    flag = generate_flag()
    known_prefix = "In cryptography, a block cipher mode of operation is an algorithm that uses a block cipher to provide information security such as confidentiality or authenticity."  # Known plaintext to assist with the attack
    plaintext = known_prefix + flag
    
    key = get_random_bytes(16)  # 128-bit key
    iv = get_random_bytes(16)   # Initialization vector for CFB mode

    cipher = AES.new(key, AES.MODE_CFB, iv=iv)
    ciphertext = cipher.encrypt(plaintext.encode('utf-8'))

    return {
        "flag": flag,
        "ciphertext": ciphertext.hex(),
        "cipher": "AES-CFB",
        "hint": "AES-CFB allows for bit-flipping attacks. Use the known prefix to manipulate the ciphertext and recover the flag.",
        "necessary_info": "The flag format is flag{...}" + "\n" + f"The plaintext starts with: {known_prefix}",
        "extra_info": f"The initialization vector (IV) is: {iv.hex()}",
        "difficulty": "hard"
    }


def generate_aes_challenge(variation):
    if variation == "aes_gcm":
        return generate_aes_gcm_challenge()
    elif variation == "aes_ccm":
        return generate_aes_ccm_challenge()
    elif variation == "aes_xts":
        return generate_aes_xts_challenge()
    elif variation == "aes_cfb":
        return generate_aes_cfb_challenge()
    else:
        raise ValueError(f"Unknown AES variation: {variation}")

# Example usage
if __name__ == "__main__":
    for mode in [
        "padding_oracle",
        "aes_gcm",
        "aes_ccm",
        "aes_xts",
        "aes_cfb",
    ]:
        try:
            challenge = generate_aes_challenge(mode)
            print(f"--- {mode.upper()} ---")
            for k, v in challenge.items():
                print(f"{k}: {v}")
            print()
        except ImportError as e:
            print(f"Skipping '{mode}' due to missing dependencies: {e}")
