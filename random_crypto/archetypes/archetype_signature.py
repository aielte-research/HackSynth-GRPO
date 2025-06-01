from crypto_helpers import generate_flag

from Crypto.Util.number import getPrime, inverse

from eth_keys import keys
from eth_utils import keccak

from ecdsa import SigningKey, SECP256k1
import hashlib
import secrets


from py_ecc.bls import G2ProofOfPossession as bls_pop
from py_ecc.optimized_bls12_381 import curve_order
import os



def generate_ecdsa_nonce_reuse_challenge():
    # Generate flag based on private key recovery
    sk = SigningKey.generate(curve=SECP256k1)
    flag = f"flag{{{sk.to_string().hex()}}}"
    vk = sk.get_verifying_key()

    # Use a fixed deterministic entropy function
    fixed_entropy = lambda n: b"\x01" * n  # Produces deterministic nonce

    msg1 = b"CTF challenge message"
    msg2 = b"Second signed message"

    # Generate signatures with nonce reuse
    sig1 = sk.sign(msg1, hashfunc=hashlib.sha256, entropy=fixed_entropy)
    sig2 = sk.sign(msg2, hashfunc=hashlib.sha256, entropy=fixed_entropy)

    return {
        "flag": flag,
        "ciphertext": f"Sig1: {sig1.hex()}\nSig2: {sig2.hex()}",  # Using ciphertext field for signatures
        "cipher": "ECDSA Nonce Reuse",
        "hint": "Identical nonce usage breaks ECDSA security. What's common between the signatures?",
        "necessary_info": (
            f"Public Key (x,y): ({vk.pubkey.point.x()}, {vk.pubkey.point.y()})\n"
            f"Messages:\n1. {msg1.decode()}\n2. {msg2.decode()}\n"
            "Curve: secp256k1"
            "The flag format is flag{hex(private_key)}"
        ),
        "difficulty": "hard"
    }


def generate_rsa_sign_with_low_public_exponent_challenge():
    flag = generate_flag()
    e = 3  # Low public exponent

    # Ensure message is small enough for cube root attack
    while True:
        # Generate 256-bit primes congruent to 2 mod 3
        p = getPrime(256)
        if p % 3 != 2:
            continue
        q = getPrime(256)
        if q % 3 != 2:
            continue
        n = p * q
        flag_int = int.from_bytes(flag.encode(), 'big')
        # Verify message fits cube root attack requirements
        if flag_int < (n ** (1/3)):
            break

    # RSA signature = m^d mod n
    phi = (p-1)*(q-1)
    d = inverse(e, phi)
    signature = pow(flag_int, d, n)

    return {
        "flag": flag,
        "ciphertext": hex(signature),
        "cipher": "RSA Low-Exponent Signature",
        "hint": "What happens when you cube small numbers? Remember e=3!",
        "necessary_info": f"Public Key (n, e): ({n}, {e})",
        "difficulty": "medium"
    }


def generate_signature_schemes_challenge(variation):
    if variation == "ecdsa_nonce_reuse":
        return generate_ecdsa_nonce_reuse_challenge()
    elif variation == "rsa_sign_with_low_public_exponent":
        return generate_rsa_sign_with_low_public_exponent_challenge()
    else:
        # Fallback for unrecognized signature scheme
        plaintext = generate_flag()
        return {
            "archetype": "signature_schemes",
            "vulnerability": variation,
            "plaintext": plaintext,
            "info": "Signature scheme not recognized or not implemented."
        }

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    for var in [
        "ecdsa_nonce_reuse",
        "rsa_sign_with_low_public_exponent",
    ]:
        print(f"--- {var.upper()} ---")
        challenge = generate_signature_schemes_challenge(var)
        for k, v in challenge.items():
            print(f"{k}: {v}")
        print()
