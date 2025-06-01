import random
import string
from math import gcd
from Crypto.Util.number import getPrime, GCD
from crypto_helpers import generate_flag


def generate_small_primes_challenge(bit_length=32):
    # 1) Generate a random "flag" string
    flag = generate_flag()
    msg_int = int.from_bytes(flag.encode("utf-8"), "big")

    # 2) Generate two small primes p and q, each ~32 bits
    p = getPrime(bit_length)
    q = getPrime(bit_length)
    n = p * q

    # 3) Compute phi(n)
    phi_n = (p - 1) * (q - 1)

    # 4) Generate a random e that is coprime with phi(n)
    #    (this is purely random, not necessarily the typical 65537).
    while True:
        e_candidate = random.randint(2, phi_n - 1)
        if GCD(e_candidate, phi_n) == 1:
            e = e_candidate
            break

    # 5) Encrypt the integer form of the flag
    ciphertext = pow(msg_int, e, n)

    # 6) Return the challenge data
    return {
        "flag": flag,
        "cipher": "RSA",
        "ciphertext": str(ciphertext),
        "vulnerability": "small primes",
        "hint": "Small primes are easily factorable with typical factoring algorithms.",
        "necessary_info": (
            "The flag format is flag{...}\n"
            f"n={n}, e={e}"
        ),
        "extra_info": f"p={p}, q={q}, φ(n)={phi_n}",
        "difficulty": "medium"
    }


def generate_repeated_prime_usage_challenge(bit_length=256):
    flag = generate_flag()
    msg_int = int.from_bytes(flag.encode("utf-8"), "big")

    # Generate a small prime p and use it twice for n = p^2
    p = getPrime(bit_length)
    q = p  # Force repeated prime
    n = p * q

    # Calculate φ(n) = (p-1)(q-1) = (p-1)^2
    phi_n = (p - 1) ** 2

    # Generate a random e that is coprime with φ(n)
    while True:
        e = random.randint(2, phi_n - 1)
        if GCD(e, phi_n) == 1:
            break

    # Encrypt the message
    ciphertext = pow(msg_int, e, n)

    return {
        "flag": flag,
        "cipher": "RSA",
        "ciphertext": str(ciphertext),
        "vulnerability": "repeated primes",
        "hint": "The RSA modulus n is a perfect square. This often indicates a repeated prime usage vulnerability.",
        "necessary_info": "The flag format is flag{...}" + "\n" + f"n={n}, e={e}",
        "extra_info": f"p={p}, q={q}, φ(n)={phi_n}",
        "difficulty": "hard"
    }


def generate_partial_key_exposure_challenge(bit_length=256):
    flag = generate_flag()
    msg_int = int.from_bytes(flag.encode("utf-8"), "big")

    # Generate two primes p and q
    p = getPrime(bit_length)
    q = getPrime(bit_length)
    n = p * q

    # Calculate φ(n) = (p-1)(q-1)
    phi_n = (p - 1) * (q - 1)

    # Generate a random e that is coprime with φ(n)
    while True:
        e = random.randint(2, phi_n - 1)
        if GCD(e, phi_n) == 1:
            break

    # Reveal the upper half of p's bits as a hint
    three_quater_bits = bit_length // 2 + bit_length // 4
    partial_p = p >> three_quater_bits  # Top three_quater_bits of p's bits

    # Encrypt the flag
    ciphertext = pow(msg_int, e, n)

    return {
        "flag": flag,
        "cipher": "RSA",
        "ciphertext": str(ciphertext),
        "vulnerability": "partial key exposure",
        "hint": (
            "A partial exposure of a prime number's bits can significantly weaken RSA security. "
            "Look into lattice-based techniques like Coppersmith's attack."
        ),
        "necessary_info": "The flag format is flag{...}" + "\n" + f"n={n}, e={e}, partial_p={partial_p}",
        "extra_info": f"p={p}, q={q}, φ(n)={phi_n}",
        "difficulty": "hard"
    }


def generate_common_factors_challenge(bit_length=256):
    flag = generate_flag()
    plaintext = "This is not the flag!"  # secondary plaintext for the second modulus

    # Convert plaintexts to integers
    msg_int1 = int.from_bytes(flag.encode("utf-8"), "big")
    msg_int2 = int.from_bytes(plaintext.encode("utf-8"), "big")

    # Generate primes with a shared factor
    p = getPrime(bit_length)
    q = getPrime(bit_length)
    r = getPrime(bit_length)

    # Two moduli sharing the prime q
    n1 = p * q
    n2 = q * r

    # Randomize the public exponent e and ensure it is coprime with φ(n1) and φ(n2)
    phi_n1 = (p - 1) * (q - 1)
    phi_n2 = (q - 1) * (r - 1)

    while True:
        e = random.randint(2, min(phi_n1, phi_n2) - 1)
        if GCD(e, phi_n1) == 1 and GCD(e, phi_n2) == 1:
            break

    # Encrypt the messages
    ciphertext1 = pow(msg_int1, e, n1)
    ciphertext2 = pow(msg_int2, e, n2)

    return {
        "flag": flag,
        "cipher": "RSA",
        "ciphertext": f"ciphertext1: {ciphertext1}, ciphertext2: {ciphertext2}",
        "vulnerability": "common factors",
        "hint": (
            "When two RSA moduli share a common factor, computing GCD(n1, n2) can "
            "reveal the shared prime factor."
        ),
        "necessary_info": "The flag format is flag{...}" + "\n" + f"n1={n1}, n2={n2}, e={e}",
        "extra_info": f"p={p}, q={q}, r={r}, φ(n1)={phi_n1}, φ(n2)={phi_n2}",
        "difficulty": "hard"
    }


def generate_shared_prime_challenge(bit_length=256):
    flag = generate_flag()
    msg_int = int.from_bytes(flag.encode("utf-8"), "big")

    # Generate primes and shared prime p
    p = getPrime(bit_length)
    q = getPrime(bit_length)
    q2 = getPrime(bit_length)

    # Create two moduli sharing the prime p
    n = p * q
    n2 = p * q2

    # Calculate φ(n) and φ(n2)
    phi_n = (p - 1) * (q - 1)
    phi_n2 = (p - 1) * (q2 - 1)

    # Generate a random e that is coprime with φ(n) and φ(n2)
    while True:
        e = random.randint(2, min(phi_n, phi_n2) - 1)
        if GCD(e, phi_n) == 1 and GCD(e, phi_n2) == 1:
            break

    # Encrypt the message using n
    ciphertext = pow(msg_int, e, n)

    return {
        "flag": flag,
        "cipher": "RSA",
        "ciphertext": str(ciphertext),
        "vulnerability": "shared prime",
        "hint": (
            "When two RSA moduli share a common prime, computing GCD(n, n2) "
            "can reveal the shared prime factor."
        ),
        "necessary_info": "The flag format is flag{...}" + "\n" + f"n={n}, n2={n2}, e={e}",
        "extra_info": f"p={p}, q={q}, q2={q2}, φ(n)={phi_n}, φ(n2)={phi_n2}",
        "difficulty": "medium"
    }


def generate_blum_integers_challenge(bit_length=256):
    def get_prime_congruent_to_3_mod_4(blen):
        while True:
            prime_candidate = getPrime(blen)
            if prime_candidate % 4 == 3:
                return prime_candidate

    flag = generate_flag()
    msg_int = int.from_bytes(flag.encode("utf-8"), "big")

    # Generate Blum primes p and q (p ≡ 3 mod 4, q ≡ 3 mod 4)
    p = get_prime_congruent_to_3_mod_4(bit_length)
    q = get_prime_congruent_to_3_mod_4(bit_length)
    n = p * q
    # Suppose we leak the lower 126 bits of p. Let's call that partial_p.
    partial_p = p & ((1 << 126) - 1)

    # Calculate φ(n) = (p-1)(q-1)
    phi_n = (p - 1) * (q - 1)

    # Generate a random e that is coprime with φ(n)
    while True:
        e = random.randint(2, phi_n - 1)
        if GCD(e, phi_n) == 1:
            break

    # Encrypt the flag using RSA
    ciphertext = pow(msg_int, e, n)

    return {
        "flag": flag,
        "cipher": "RSA",
        "ciphertext": str(ciphertext),
        "vulnerability": "blum integers",
        "hint": (
            "Blum integers are products of primes congruent to 3 modulo 4. "
            "These integers have unique properties in cryptography."
        ),
        "necessary_info": "The flag format is flag{...}" + "\n" + f"n={n}, e={e}" + "\n" + f"The lower 104 bits of p are: {partial_p}\n",
        "extra_info": f"p={p}, q={q}, φ(n)={phi_n}",
        "difficulty": "hard"
    }


def generate_rsa_challenge(variation):
    # Randomly select a 'default' bit length if not small primes
    random_bit_length = random.choice([256, 384, 512])

    if variation == "small_primes":
        # Forcibly smaller bit_length (128)
        return generate_small_primes_challenge()

    elif variation == "repeated_prime_usage":
        return generate_repeated_prime_usage_challenge(bit_length=random_bit_length)

    elif variation == "partial_key_exposure":
        return generate_partial_key_exposure_challenge(bit_length=256)

    elif variation == "common_factors":
        return generate_common_factors_challenge(bit_length=random_bit_length)

    elif variation == "shared_prime":
        return generate_shared_prime_challenge(bit_length=random_bit_length)

    elif variation == "blum_integers":
        return generate_blum_integers_challenge(bit_length=256)

    else:
        raise ValueError(f"Unknown RSA variation: {variation}")

# Example Usage
if __name__ == "__main__":
    for var in [
        "small_primes", 
        "repeated_prime_usage", 
        "partial_key_exposure",
        "common_factors", 
        "shared_prime", 
        "blum_integers"
    ]:
        challenge = generate_rsa_challenge(var)
        print(f"--- {var.upper()} ---")
        for k, v in challenge.items():
            print(f"{k}: {v}")
        print()
