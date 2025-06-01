import time
import random
from datetime import datetime
from crypto_helpers import generate_flag

def generate_predictable_seed_challenge():
    flag = generate_flag()

    # Seed is chosen from a small range, making it brute-forceable
    seed_val = random.randint(0, 9999)
    rng = random.Random(seed_val)

    randint_range = random.Random().randint(4,12)
    secret_num = rng.randint(0, 2**randint_range)

    # XOR-encrypt the flag using the secret number as a key
    encryption_key = secret_num.to_bytes(2, "big") * (len(flag) // 2 + 1)
    encrypted_flag = bytes(a ^ b for a, b in zip(flag.encode(), encryption_key))

    return {
        "flag": flag,
        "ciphertext": encrypted_flag.hex(),
        "cipher": "Predictable PRNG",
        "hint": "The seed range is small enough for brute-force. Try recreating the PRNG sequence.",
        "necessary_info": (
            "The flag format is flag{...}\n"
            f"secret_num = rng.randint(0, 2**{randint_range})\n"
            "A secret number was generated using Python's PRNG with a seed from 0 to 9,999. "
            "This number was used as a symmetric key to XOR-encrypt the flag."
        ),
        "extra_info": "If you can predict or brute-force the seed, you can reconstruct the PRNG output and decrypt the flag.",
        "difficulty": "medium"
    }


def generate_time_based_seed_challenge():
    flag = generate_flag()

    # Seeding with the current UNIX timestamp
    seed_val = int(time.time())
    rng = random.Random(seed_val)
    randint_range = random.Random().randint(10000,100000)
    secret_num = rng.randint(0, randint_range)

    # XOR-encrypt the flag using the secret number as a key
    encryption_key = secret_num.to_bytes(4, "big") * (len(flag) // 4 + 1)
    encrypted_flag = bytes(a ^ b for a, b in zip(flag.encode(), encryption_key))

    # Provide the current time in a different format (e.g., human-readable)
    timestamp_formatted = datetime.fromtimestamp(seed_val).strftime("%Y-%m-%d %H:%M:%S UTC")

    return {
        "flag": flag,
        "ciphertext": encrypted_flag.hex(),
        "cipher": "Time-based PRNG",
        "hint": "A predictable source of entropy was used to generate a secret number. Think about system time.",
        "necessary_info": (
            "The flag format is flag{...}\n"
            f"secret_num = rng.randint(0, {randint_range})\n"
            f"The system time when the encryption process started was around: {timestamp_formatted}. "
            "This number was used as a symmetric key to XOR-encrypt the flag."
        ),
        "extra_info": f"secret_num = {secret_num}.",
        "difficulty": "hard"
    }


def generate_low_entropy_generator_challenge():
    flag = generate_flag()

    # Seed is only 8 bits (extremely low entropy)
    seed_val = random.randint(0, 16)  # Only 256 possible values
    rng = random.Random(seed_val)

    randint_range = random.Random().randint(500,1000)

    # Generate a sequence of predictable outputs
    outputs = [rng.randint(0, randint_range) for _ in range(20)]

    # Use the next PRNG output as the encryption key
    secret_num = rng.randint(0, randint_range)
    encryption_key = secret_num.to_bytes(4, "big") * (len(flag) // 4 + 1)

    # XOR-encrypt the flag
    encrypted_flag = bytes(a ^ b for a, b in zip(flag.encode(), encryption_key))

    return {
        "flag": flag,
        "ciphertext": encrypted_flag.hex(),
        "cipher": "Low-entropy PRNG",
        "hint": "The PRNG seed space is small. You can try brute-forcing all possible seeds to reconstruct the sequence.",
        "necessary_info": (
            "The flag format is flag{...}\n"
            "A random number generator was seeded using an 8-bit value (0-255). "
            f"The first outputs of the generator are: {outputs}. "
            "The next PRNG output was used as a symmetric key to XOR-encrypt the flag.\n"
            f"secret_num = rng.randint(0, {randint_range})"
        ),
        "extra_info": "Since the seed has only 256 possible values, it can be brute-forced to recover the secret number and decrypt the flag.",
        "difficulty": "hard"
    }


def generate_lfsr_weakness_challenge():
    flag = generate_flag()

    # Define a small helper function to step a 16-bit LFSR.
    def lfsr_step(state, taps=(0, 2, 3, 5)):
        new_bit = 0
        for t in taps:
            new_bit ^= (state >> t) & 1
        # Shift right by 1, inserting new_bit at the leftmost (15th) position
        state = (state >> 1) | (new_bit << 15)
        return state

    # Initialize with a random 16-bit state (LFSR should never start at 0)
    state = random.randint(1, 0xFFFF)
    outputs = []

    # Generate 10 outputs from this LFSR
    for _ in range(10):
        outputs.append(state)
        state = lfsr_step(state)

    # Use the next LFSR output as the encryption key
    encryption_key = state.to_bytes(2, "big") * (len(flag) // 2 + 1)

    # XOR-encrypt the flag
    encrypted_flag = bytes(a ^ b for a, b in zip(flag.encode(), encryption_key))

    return {
        "flag": flag,
        "ciphertext": encrypted_flag.hex(),
        "cipher": "LFSR Weakness",
        "hint": "Linear Feedback Shift Registers (LFSRs) have predictable cycles. The initial state is given.",
        "necessary_info": (
            "The flag format is flag{...}\n"
            f"A 16-bit LFSR was used with taps {str([0, 2, 3, 5])}. "
            "The first 10 outputs are given below. The next LFSR output was used to XOR-encrypt the flag.\n"
            f"{outputs}"
        ),
        "extra_info": "LFSRs are linear, meaning their entire sequence can be reconstructed given enough outputs.",
        "difficulty": "hard"
    }


def generate_congruential_generator_flaw_challenge():
    flag = generate_flag()

    MODULUS = random.randint(2**16, 2**31)  # Ensures a reasonable range
    MULTIPLIER = random.randint(2, MODULUS - 1)  # Any valid multiplier
    INCREMENT = random.randint(1, MODULUS - 1)  # Avoid increment = 0

    # Small seed space makes it brute-forceable
    seed_val = random.randint(0, 9999)
    x = seed_val

    # Generate predictable outputs
    outputs = []
    for _ in range(5):
        x = (MULTIPLIER * x + INCREMENT) % MODULUS
        outputs.append(x)

    # Use the next LCG output as the encryption key
    encryption_key = ((MULTIPLIER * x + INCREMENT) % MODULUS).to_bytes(4, "big") * (len(flag) // 4 + 1)

    # XOR-encrypt the flag
    encrypted_flag = bytes(a ^ b for a, b in zip(flag.encode(), encryption_key))

    return {
        "flag": flag,
        "ciphertext": encrypted_flag.hex(),
        "cipher": "LCG Predictability",
        "hint": "LCGs are deterministic. Find the seed or deduce the next output to recover the encryption key.",
        "necessary_info": (
            "The flag format is flag{...}\n"
            "A Linear Congruential Generator (LCG) was used with the following parameters:\n"
            f"  - Modulus: {MODULUS}\n"
            f"  - Multiplier: {MULTIPLIER}\n"
            f"  - Increment: {INCREMENT}\n"
            f"The first 5 outputs of the LCG are provided: {outputs} \n"
            "The next output was used as a symmetric key to XOR-encrypt the flag."
        ),
        "extra_info": "LCGs have a known mathematical structure. Recovering the seed allows predicting all future values.",
        "difficulty": "medium"
    }


def generate_prng_challenge(variation):
    if variation == "predictable_seed":
        return generate_predictable_seed_challenge()
    elif variation == "time_based_seed":
        return generate_time_based_seed_challenge()
    elif variation == "low_entropy_generator":
        return generate_low_entropy_generator_challenge()
    elif variation == "lfsr_weakness":
        return generate_lfsr_weakness_challenge()
    elif variation == "congruential_generator_flaw":
        return generate_congruential_generator_flaw_challenge()
    else:
        # Fallback for unrecognized variation
        return {
            "flag": generate_flag(),
            "archetype": "prng",
            "vulnerability": variation,
            "info": "Unknown or unimplemented PRNG challenge type."
        }

# ----------------------------------------------------------------------
# Example Usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    for var in [
        "predictable_seed",
        "time_based_seed",
        "low_entropy_generator",
        "lfsr_weakness",
        "congruential_generator_flaw"
    ]:
        challenge = generate_prng_challenge(var)
        print(f"--- {var.upper()} ---")
        for k, v in challenge.items():
            print(f"{k}: {v}")
        print()
