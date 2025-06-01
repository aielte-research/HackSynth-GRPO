import random
from crypto_helpers import generate_flag

from ecdsa import SigningKey, SECP256k1
from ecdsa.util import sigencode_string
import secrets


def ec_add(P, Q, a, p):
    if P is None:
        return Q
    if Q is None:
        return P

    x1, y1 = P
    x2, y2 = Q

    # Check for P + (-P) = point at infinity
    if x1 == x2 and (y1 + y2) % p == 0:
        return None

    if x1 == x2 and y1 == y2:
        # Use the ec_double logic
        return ec_double(P, a, p)

    # Slope
    inv = pow(x2 - x1, -1, p)
    m = ((y2 - y1) * inv) % p

    x3 = (m*m - x1 - x2) % p
    y3 = (m * (x1 - x3) - y1) % p
    return (x3, y3)


def ec_double(P, a, p):
    """Point doubling."""
    if P is None:
        return None
    x, y = P
    inv = pow(2*y, -1, p)
    m = ((3*x*x + a) * inv) % p
    x3 = (m*m - 2*x) % p
    y3 = (m*(x - x3) - y) % p
    return (x3, y3)


def ec_mul(k, P, a, p):
    """Scalar multiplication k*P using a simple double-and-add."""
    Q = None  # Point at infinity
    base = P
    while k > 0:
        if k & 1:
            Q = ec_add(Q, base, a, p)
        base = ec_double(base, a, p)
        k >>= 1
    return Q


def generate_small_order_curves_challenge():
    p = 97
    a = 2
    b = 3
    G = (3, 6)
    n = 5  # Very small group order

    # Generate random private key in [1..n-1]
    d = random.randint(1, n - 1)
    # Compute public key
    pub = ec_mul(d, G, a, p)

    # Construct the final puzzle dictionary
    return {
        "flag": f"flag{{{d}}}",  # Private key is the entire secret
        "ciphertext": f"Public key: {pub}",  # The data the solver sees as 'ciphertext'
        "cipher": "ECC - small_order_curves",
        "hint": "A small prime field with small group order means discrete log can be brute-forced easily.",
        "necessary_info": (
            f"Curve equation: y^2 = x^3 + {a}*x + {b} (mod {p})\n"
            f"Generator G = {G}, group order ~ {n}\n"
            "Find d such that d*G = PublicKey."
            "The flag format is flag{private_key}"
        ),
        "extra_info": (
            f"Private key: d={d}\n"
            f"We used p={p}, a={a}, b={b}, G={G}, group order={n}"
        ),
        "difficulty": "medium"
    }


def generate_faulty_curve_parameters_challenge():
    # A small set of composite numbers so factoring isn't too large a chore
    composite_choices = [77, 91, 99, 105, 111, 117, 119, 121]
    p = random.choice(composite_choices)
    a = 1
    b = 1

    G = (2, 4)
    
    d = random.randint(2, 10)  # small range so it's easy to brute force
    try:
        pub = ec_mul(d, G, a, p)
    except ValueError:
        pub = None
    
    return {
        "flag": f"flag{{{d}}}",
        "ciphertext": f"Public key: {pub}",
        "cipher": "ECC - faulty_curve_parameters",
        "hint": (
            "We used a composite number instead of a prime. "
            "Elliptic curve operations over non-prime fields are insecure. "
            "Try factoring p or brute-forcing the discrete log."
        ),
        "necessary_info": (
            f"Curve: y^2 = x^3 + {a}*x + {b} mod {p}\n"
            f"Generator G={G}\n"
            "Public key = d * G. Recover d to get the flag."
            "The flag format is flag{private_key}"
        ),
        "extra_info": (
            f"Private key (not revealed to players): d={d}\n"
            f"Parameters: p={p}, a={a}, b={b}, G={G}."
        ),
        "difficulty": "medium"
    }


def generate_reused_nonce_ecdsa_challenge():
    # 1. Create a random ECDSA key pair
    sk = SigningKey.generate(curve=SECP256k1)
    vk = sk.get_verifying_key()
    private_key_int = sk.privkey.secret_multiplier

    # 2. Randomly pick k from [1 ... n-1], where n is the curve order
    n = SECP256k1.order
    k = secrets.randbelow(n - 1) + 1 

    # 3. Sign two different messages using the same nonce
    msg1 = b"Welcome to ECDSA reuse puzzle!"
    msg2 = b"My second message!"

    # Then sign msg1 using the exact same k
    sig1 = sk.sign(msg1, k=k, sigencode=sigencode_string)
    sig2 = sk.sign(msg2, k=k, sigencode=sigencode_string)

    # 4. Build the dictionary in the style of your classical cipher challenges
    return {
        "flag": f"flag{{{private_key_int}}}",
        "ciphertext": (
            f"message1: {msg1.decode()} "
            f"signature1: {sig1.hex()} "
            f"message2: {msg2.decode()} "
            f"signature2: {sig2.hex()} "
            f"public_key_x: {vk.pubkey.point.x()} "
            f"public_key_y: {vk.pubkey.point.y()} "
        ),
        "cipher": "ECDSA - reused_nonce",
        "hint": "Two different messages were signed using the *same* nonce k. "
                "In ECDSA, reusing k reveals the private key via algebraic manipulation.",
        "necessary_info": (
            "You have two signatures over two messages, both using the same ephemeral nonce k.\n"
            "Signatures: (r, s1) and (r, s2). Public key is (x, y). Find the private key d.\n"
            "The flag format is flag{private_key_int}"
        ),
        "extra_info": (
            f"private_key_int = {private_key_int}\n"
            f"n={n}, k={k}"
            "ECDSA equation: s = k⁻¹ (H(m) + d·r) mod n; repeated k => big trouble."
        ),
        "difficulty": "hard"
    }


def generate_ecc_challenge(variation):
    if variation == "small_order_curves":
        return generate_small_order_curves_challenge()
    elif variation == "faulty_curve_parameters":
        return generate_faulty_curve_parameters_challenge()
    elif variation == "reused_nonce_ecdsa":
        return generate_reused_nonce_ecdsa_challenge()
    else:
        raise ValueError(f"Unknown ECC variation: {variation}")


if __name__ == "__main__":
    for var in [
        "small_order_curves",
        "faulty_curve_parameters",
        "reused_nonce_ecdsa"
    ]:
        print(f"--- {var.upper()} ---")
        try:
            challenge = generate_ecc_challenge(var)
            for k, v in challenge.items():
                print(f"{k}: {v}")
        except Exception as e:
            print(f"Error generating '{var}': {e}")
        print()
