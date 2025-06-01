import random
import string
from math import gcd
from crypto_helpers import generate_flag
import base64
import os


def generate_caesar_challenge():
    flag = generate_flag()
    shift = random.randint(1, 25)

    encrypted_chars = []
    for ch in flag:
        if ch.isalpha():
            # Always handle as lowercase for simplicity
            base = ord('a')
            # Convert to 0-25 range, add shift, wrap mod 26, convert back
            new_ord = (ord(ch.lower()) - base + shift) % 26 + base
            encrypted_chars.append(chr(new_ord))
        else:
            encrypted_chars.append(ch)

    ciphertext = "".join(encrypted_chars)
    return {
        "flag": flag,
        "ciphertext": ciphertext,
        "cipher": "Caesar",
        "hint": "Caesar ciphers are easily broken by brute force or frequency analysis.",
        "necessary_info": "The flag format is flag{...}",
        "extra_info": f"The shift used is: {shift}",
        "difficulty": "easy",
    }


def generate_vigenere_challenge():
    flag = generate_flag()
    key_length = random.randint(4, 8)
    key = ''.join(random.choices(string.ascii_lowercase, k=key_length))

    encrypted_chars = []
    key_index = 0
    for ch in flag:
        if ch.isalpha():
            base = ord('a')
            shift = ord(key[key_index % len(key)]) - base
            new_ord = (ord(ch.lower()) - base + shift) % 26 + base
            encrypted_chars.append(chr(new_ord))
            key_index += 1
        else:
            encrypted_chars.append(ch)

    ciphertext = "".join(encrypted_chars)
    return {
        "flag": flag,
        "ciphertext": ciphertext,
        "cipher": "Vigenère",
        "hint": "Vigenère ciphers are vulnerable to Kasiski examination and frequency analysis.",
        "necessary_info": "The flag format is flag{...}" + "\n" + f"The keyword is: {key}",
        "difficulty": "medium"
    }


def generate_rail_fence_challenge():
    flag = generate_flag()
    rails = random.randint(2, 3)

    lines = [[] for _ in range(rails)]
    rail_idx = 0
    direction = 1

    for ch in flag:
        lines[rail_idx].append(ch)
        rail_idx += direction
        if rail_idx == rails - 1:
            direction = -1
        elif rail_idx == 0:
            direction = 1

    ciphertext = "".join("".join(line) for line in lines)

    return {
        "flag": flag,
        "ciphertext": ciphertext,
        "cipher": "Rail Fence",
        "hint": "Rail Fence is a simple transposition cipher vulnerable to reconstruction by analyzing positions.",
        "necessary_info": "The flag format is flag{...}",
        "extra_info": f"The rails value is: {rails}",
        "difficulty": "hard"
    }


def generate_playfair_challenge():
    def build_playfair_table(key_phrase):
        # Remove duplicates, treat 'j' as 'i'
        processed_key = ""
        for c in key_phrase.lower():
            if c == 'j':
                c = 'i'
            if c not in processed_key and c.isalpha():
                processed_key += c

        # Fill in the remaining letters of the alphabet (no 'j')
        alphabet = "abcdefghiklmnopqrstuvwxyz"
        for c in alphabet:
            if c not in processed_key:
                processed_key += c

        table = [list(processed_key[i*5:(i+1)*5]) for i in range(5)]
        return table

    # Prepare plaintext function
    def prep_text(txt):
        txt = txt.lower()
        txt = ''.join(ch for ch in txt if ch.isalpha())
        txt = txt.replace('j', 'i')
        return txt

    flag = generate_flag()
    extra_text = "The Playfair cipher or Playfair square or Wheatstone–Playfair cipher is a manual symmetric encryption technique and was the first literal digram substitution cipher."
    key_phrase = ''.join(random.choices(string.ascii_lowercase, k=8))
    table = build_playfair_table(key_phrase)
    
    # Build a dict for letter positions in the table
    positions = {}
    for r in range(5):
        for c in range(5):
            positions[table[r][c]] = (r, c)

    # Prepare the flag for encryption
    prep = prep_text(extra_text + flag)

    # Break into digraphs, insert 'x' if needed
    digraphs = []
    i = 0
    while i < len(prep):
        c1 = prep[i]
        c2 = ''
        if i + 1 < len(prep):
            c2 = prep[i+1]
        else:
            c2 = 'x'

        if c1 == c2:
            digraphs.append((c1, 'x'))
            i += 1
        else:
            digraphs.append((c1, c2))
            i += 2

    # Encrypt each pair
    encrypted_pairs = []
    for a, b in digraphs:
        r1, c1 = positions[a]
        r2, c2 = positions[b]

        if r1 == r2:
            # Same row => shift right
            c1 = (c1 + 1) % 5
            c2 = (c2 + 1) % 5
        elif c1 == c2:
            # Same column => shift down
            r1 = (r1 + 1) % 5
            r2 = (r2 + 1) % 5
        else:
            # Rectangle => swap columns
            c1, c2 = c2, c1

        encrypted_pairs.append(table[r1][c1] + table[r2][c2])

    ciphertext = "".join(encrypted_pairs)

    return {
        "flag": flag,
        "ciphertext": ciphertext,
        "cipher": "Playfair",
        "hint": "Playfair is vulnerable to digraph frequency analysis.",
        "necessary_info": "The flag format is flag{...}" + "\n" + f"The keyword is: {key_phrase}",
        "extra_info": f"The playfair table is {table}",
        "difficulty": "hard"
    }


def generate_hill_challenge():
    def char_to_num(c):
        return ord(c) - ord('a')

    def num_to_char(n):
        return chr(n + ord('a'))

    def prep_text(txt):
        txt = txt.lower()
        txt = ''.join(ch for ch in txt if ch.isalpha())
        return txt

    def random_invertible_matrix_2x2():
        while True:
            a, b, c, d = [random.randint(0, 25) for _ in range(4)]
            det = a*d - b*c
            # Must be invertible mod 26 => gcd(det, 26) == 1
            if gcd(det, 26) == 1:
                return [[a, b], [c, d]]

    flag = generate_flag(numeric=False)
    known_start = "welcome to the ancient library. hidden within these words is your key: "
    prep = prep_text(known_start + flag)
    # Pad to even length
    if len(prep) % 2 != 0:
        prep += 'x'

    matrix = random_invertible_matrix_2x2()

    encrypted_chars = []
    for i in range(0, len(prep), 2):
        pair = prep[i:i+2]
        x1 = char_to_num(pair[0])
        x2 = char_to_num(pair[1])

        y1 = (matrix[0][0] * x1 + matrix[0][1] * x2) % 26
        y2 = (matrix[1][0] * x1 + matrix[1][1] * x2) % 26

        encrypted_chars.append(num_to_char(y1))
        encrypted_chars.append(num_to_char(y2))

    ciphertext = "".join(encrypted_chars)

    return {
        "flag": flag,
        "ciphertext": ciphertext,
        "cipher": "Hill",
        "hint": "Hill ciphers can be broken using known-plaintext attacks and linear algebra mod 26.",
        "necessary_info": "The flag format is flag{...}" + "\n" + f"You got a tip from a friend that the encrypted message starts with: {known_start}",
        "extra_info": f"The matrix used is: {matrix}",
        "difficulty": "hard"
    }


def generate_substitution_challenge():
    flag = generate_flag()

    extra_text = 'In cryptography, a substitution cipher is a method of encrypting in which units of plaintext are replaced with the ciphertext, in a defined manner, with the help of a key; the "units" may be single letters (the most common), pairs of letters, triplets of letters, mixtures of the above, and so forth. The receiver deciphers the text by performing the inverse substitution process to extract the original message. Likewise, Jacques quickly zipped through a question on zero-knowledge proofs, ensuring no critical letter was left out of our cryptographic puzzle.'
    
    alph = list(string.ascii_lowercase)
    random.shuffle(alph)
    substitution_map = {chr(ord('a') + i): alph[i] for i in range(26)}

    encrypted_chars = []
    for ch in extra_text + flag:
        if ch.isalpha():
            encrypted_chars.append(substitution_map[ch.lower()])
        else:
            encrypted_chars.append(ch)

    ciphertext = "".join(encrypted_chars)

    return {
        "flag": flag,
        "ciphertext": ciphertext,
        "cipher": "Substitution",
        "hint": "Simple substitution ciphers are susceptible to frequency analysis.",
        "necessary_info": "The flag format is flag{...}",
        "extra_info": f"The substitution map is: {substitution_map}",
        "difficulty": "hard"
    }


def generate_substitution_direct_challenge():
    flag = generate_flag()

    alph = list(string.ascii_lowercase)
    shuffled = alph[:]  # Make a copy of the alphabet
    random.shuffle(shuffled)
    substitution_map = dict(zip(alph, shuffled))

    ciphertext_chars = []
    for ch in flag:
        if ch.lower() in substitution_map:
            if ch.isupper():
                ciphertext_chars.append(substitution_map[ch.lower()].upper())
            else:
                ciphertext_chars.append(substitution_map[ch])
        else:
            ciphertext_chars.append(ch)

    ciphertext = "".join(ciphertext_chars)

    return {
        "flag": flag,
        "cipher": "Substitution",
        "ciphertext": ciphertext,
        "hint": "Use the provided substitution table to restore each letter of the ciphertext to the original plaintext.",
        "necessary_info": (
            "The flag format is flag{...}\n\n"
            "Below is the substitution table mapping plaintext letters to ciphertext letters:\n"
            f"{substitution_map}"
        ),
        "extra_info": "Letters outside [a-zA-Z] are left unchanged. Case is preserved by converting letters back to uppercase if needed.",
        "difficulty": "medium",
    }


def generate_transposition_challenge():
    flag = generate_flag()
    extra_text = "In cryptography, a transposition cipher (also known as a permutation cipher) is a method of encryption which scrambles the positions of characters (transposition) without changing the characters themselves. Transposition ciphers reorder units of plaintext (typically characters or groups of characters) according to a regular system to produce a ciphertext which is a permutation of the plaintext. They differ from substitution ciphers, which do not change the position of units of plaintext but instead change the units themselves. Despite the difference between transposition and substitution operations, they are often combined, as in historical ciphers like the ADFGVX cipher or complex high-quality encryption methods like the modern Advanced Encryption Standard (AES)."
    prep = extra_text.replace(" ", "") + flag.replace(" ", "")  # optional removal of spaces

    # Generate a random key (permutation of length k)
    k = random.randint(4, 6)
    key_order = list(range(k))
    random.shuffle(key_order)

    # Break plaintext into rows of length k
    rows = []
    for i in range(0, len(prep), k):
        rows.append(list(prep[i:i+k]))

    # Pad the last row if needed
    if len(rows[-1]) < k:
        rows[-1] += ['_'] * (k - len(rows[-1]))

    # Read columns in the order of key_order
    encrypted_cols = []
    for col_index in key_order:
        for row in rows:
            encrypted_cols.append(row[col_index])

    ciphertext = "".join(encrypted_cols)

    return {
        "flag": flag,
        "ciphertext": ciphertext,
        "cipher": "Transposition",
        "hint": "Columnar transposition ciphers can be broken by analyzing column lengths and patterns.",
        "necessary_info": "The flag format is flag{...}",
        "extra_info": f"The key length is {k}, and the key order is {key_order}.",
        "difficulty": "medium"
    }


def generate_autokey_challenge():
    def autokey_encrypt(pt, seed):
        pt = pt.lower()
        # full_key = seed + plaintext
        full_key = seed + pt
        encrypted_chars = []
        ki = 0
        for ch in pt:
            if ch.isalpha():
                shift = ord(full_key[ki]) - ord('a')
                base = ord('a')
                enc_val = (ord(ch) - base + shift) % 26
                encrypted_chars.append(chr(base + enc_val))
                ki += 1
            else:
                encrypted_chars.append(ch)
        return "".join(encrypted_chars)

    flag = generate_flag()
    known_start = "In each case, the resulting plaintext appears almost random."
    seed_length = random.randint(3,4)
    key_seed = ''.join(random.choices(string.ascii_lowercase, k=seed_length))

    ciphertext = autokey_encrypt(known_start + flag, key_seed)

    return {
        "flag": flag,
        "ciphertext": ciphertext,
        "cipher": "Autokey",
        "hint": "Autokey ciphers are vulnerable once enough plaintext is known, allowing reconstruction of the key.",
        "necessary_info": "The flag format is flag{...}" + "\n" + f"You got a tip from a friend that the encrypted message starts with: {known_start}" + "\n" + f"The seed lenght is {seed_length} and only alphabetic characters count.",
        "extra_info": f"The key seed is: {key_seed}",
        "difficulty": "hard"
    }


def generate_morse_code_challenge():
    morse_code_dict = { 'A':'.-', 'B':'-...',
                    'C':'-.-.', 'D':'-..', 'E':'.',
                    'F':'..-.', 'G':'--.', 'H':'....',
                    'I':'..', 'J':'.---', 'K':'-.-',
                    'L':'.-..', 'M':'--', 'N':'-.',
                    'O':'---', 'P':'.--.', 'Q':'--.-',
                    'R':'.-.', 'S':'...', 'T':'-',
                    'U':'..-', 'V':'...-', 'W':'.--',
                    'X':'-..-', 'Y':'-.--', 'Z':'--..',
                    '1':'.----', '2':'..---', '3':'...--',
                    '4':'....-', '5':'.....', '6':'-....',
                    '7':'--...', '8':'---..', '9':'----.',
                    '0':'-----', ', ':'--..--', '.':'.-.-.-',
                    '?':'..--..', '/':'-..-.', '-':'-....-',
                    '(':'-.--.', ')':'-.--.-'}
    
    flag = generate_flag()
    
    ciphertext = ' '.join(morse_code_dict[ch.upper()] if ch.upper() in morse_code_dict else ch for ch in flag)
    
    return {
        "flag": flag,
        "ciphertext": ciphertext,
        "cipher": "Morse Code",
        "hint": "Morse Code represents each letter as a series of dots and dashes. Spaces separate letters, and slashes separate words.",
        "necessary_info": "The flag format is flag{...}",
        "extra_info": "Morse Code follows international standards, with letters separated by spaces and words separated by slashes.",
        "difficulty": "medium",
    }


def generate_fibonacci_encoding_challenge():
    def generate_fibonacci(n):
        """Generate the first N Fibonacci numbers starting from 1, 2."""
        fib = [1, 2]  
        while len(fib) < n:
            fib.append(fib[-1] + fib[-2])
        return fib

    def fibonacci_encoding(num, fib_sequence):
        """Convert a number to its Fibonacci Encoding representation."""
        encoding = []
        for fib in reversed(fib_sequence):
            if fib <= num:
                encoding.append("1")
                num -= fib
            else:
                encoding.append("0")

        encoding = encoding[encoding.index("1"):]  # Remove leading zeros
        encoding.append("1")  # Fibonacci Encoding convention
        return "".join(encoding)

    # Generate Fibonacci numbers for encoding
    fib_sequence = generate_fibonacci(20)

    # Generate flag and encode it
    flag = generate_flag()
    encoded_flag = " ".join(fibonacci_encoding(ord(ch), fib_sequence) for ch in flag)

    return {
        "flag": flag,
        "ciphertext": encoded_flag,
        "cipher": "Fibonacci Encoding",
        "hint": "Each ASCII character is represented as a sum of non-consecutive Fibonacci numbers, always ending in 1.",
        "necessary_info": "The flag format is flag{...}",
        "extra_info": "Each character of the flag is independently encoded using Fibonacci representation.",
        "difficulty": "medium"
    }


def generate_xor_challenge():
    flag = generate_flag()
    
    key = ''.join(random.choices(string.ascii_letters + string.digits, k=len(flag)))
    
    xored_bytes = []
    for f_char, k_char in zip(flag, key):
        xored_bytes.append(ord(f_char) ^ ord(k_char))
    
    xored_hex = ''.join(f"{b:02x}" for b in xored_bytes)
    key_hex = ''.join(f"{ord(c):02x}" for c in key)
    
    return {
        "flag": flag,
        "cipher": "XOR",
        "ciphertext": (
            f"stringA: {xored_hex} "
            f"stringB: {key_hex}"
        ),
        "hint": "XOR is undone by XORing again with the same key. (A XOR B = C ⇒ C XOR B = A)",
        "necessary_info": "The flag format is flag{...}",
        "extra_info": "Both strings are in hex. XOR the two provided hex strings to recover the flag.",
        "difficulty": "easy",
    }


def generate_base64_decode_challenge():
    flag = generate_flag()
    
    b64_encoded_flag = base64.b64encode(flag.encode()).decode()
    
    return {
        "flag": flag,
        "cipher": "Base64",
        "ciphertext": b64_encoded_flag,
        "hint": "Use a Base64 decoder to obtain the original text.",
        "necessary_info": "The flag format is flag{...}",
        "extra_info": "Decoding is direct; no additional layers or tricks.",
        "difficulty": "easy",
    }


def generate_base64_layered_challenge():
    flag = generate_flag()

    num_layers = random.randint(5, 10)

    encoded_bytes = flag.encode()  # Convert to bytes
    for _ in range(num_layers):
        encoded_bytes = base64.b64encode(encoded_bytes)

    layered_b64 = encoded_bytes.decode()

    return {
        "flag": flag,
        "cipher": "Base64",
        "ciphertext": layered_b64,
        "hint": "The message is encoded multiple times with Base64. Decode iteratively to recover the original flag.",
        "necessary_info": "The flag format is flag{...}",
        "extra_info": f"Encoded {num_layers} times in total.",
        "difficulty": "easy",
    }


def generate_base85_decode_challenge():
    flag = generate_flag()

    base85_encoded_flag = base64.b85encode(flag.encode()).decode("ascii")

    return {
        "flag": flag,
        "cipher": "Base85",
        "ciphertext": base85_encoded_flag,
        "hint": "Use a Base85 decoder to retrieve the original text.",
        "necessary_info": "The flag format is flag{...}",
        "extra_info": "Decoding is straightforward; only one layer is used.",
        "difficulty": "easy",
    }


def generate_base85_layered_challenge():
    flag = generate_flag()

    num_layers = random.randint(5, 10)

    encoded_bytes = flag.encode()
    for _ in range(num_layers):
        encoded_bytes = base64.b85encode(encoded_bytes)

    layered_b85 = encoded_bytes.decode("ascii")

    return {
        "flag": flag,
        "cipher": "Base85",
        "ciphertext": layered_b85,
        "hint": "The message is encoded multiple times with Base85. Decode iteratively to recover the original flag.",
        "necessary_info": "The flag format is flag{...}",
        "extra_info": f"Encoded {num_layers} times in total.",
        "difficulty": "easy",
    }


def generate_atbash_challenge():
    flag = generate_flag()
    alph = "abcdefghijklmnopqrstuvwxyz"
    reversed_alph = alph[::-1]  # "zyxwvutsrqponmlkjihgfedcba"
    
    # Build a dictionary for both lowercase and uppercase
    atbash_map = {}
    for a, r in zip(alph, reversed_alph):
        atbash_map[a] = r
        atbash_map[a.upper()] = r.upper()

    def atbash_char(c):
        return atbash_map[c] if c in atbash_map else c

    ciphertext = "".join(atbash_char(ch) for ch in flag)

    return {
        "flag": flag,
        "cipher": "Atbash",
        "ciphertext": ciphertext,
        "hint": "Atbash reverses the alphabet (A ↔ Z, B ↔ Y, etc.).",
        "necessary_info": "The flag format is flag{...}",
        "difficulty": "easy",
    }


def generate_hex_challenge():
    flag = generate_flag()
    ciphertext = flag.encode().hex()  # e.g., '68656c6c6f' for "hello"
    
    return {
        "flag": flag,
        "cipher": "Hex",
        "ciphertext": ciphertext,
        "hint": "Hex encoding represents bytes as pairs of 0-9 and a-f.",
        "necessary_info": "The flag format is flag{...}",
        "difficulty": "easy",
    }


def generate_ascii_shift_challenge():
    flag = generate_flag()
    shift = random.randint(1, 10)

    shifted_chars = []
    for ch in flag:
        new_val = (ord(ch) + shift) % 256
        shifted_chars.append(chr(new_val))
    
    ciphertext = "".join(shifted_chars)
    
    return {
        "flag": flag,
        "cipher": "ASCII Shift",
        "ciphertext": ciphertext,
        "hint": f"The ASCII values of each character have been shifted by some small amount (1–10).",
        "necessary_info": "The flag format is flag{...}",
        "extra_info": f"Shift used: {shift}",
        "difficulty": "easy",
    }


def generate_split_number_challenge():
    flag = generate_flag(numeric=False)  # Ensure only letters inside the flag

    # Random split into 2–4 parts
    num_parts = random.randint(2, 4)
    split_indices = sorted(random.sample(range(1, len(flag)), num_parts - 1))
    parts = []
    last = 0
    for idx in split_indices:
        parts.append(flag[last:idx])
        last = idx
    parts.append(flag[last:])

    separator = " " + str(random.randint(0,9)) * random.randint(5, 9) + " "
    ciphertext = separator + separator.join(parts) + separator
    return {
        "flag": flag,
        "ciphertext": ciphertext,
        "cipher": "SplitNumberMix",
        "hint": "Numbers are used to break up the flag. The original flag uses only letters.",
        "necessary_info": "The flag format is flag{...}",
        "extra_info": f"Split into {num_parts} parts with numeric sequences of length 2–5 randomly inserted.",
        "difficulty": "easy",
    }


def generate_reversal_challenge():
    flag = generate_flag()
    reversed_flag = flag[::-1]

    return {
        "flag": flag,
        "ciphertext": reversed_flag,
        "cipher": "Reverse",
        "hint": "The original message has simply been reversed.",
        "necessary_info": "The flag format is flag{...}",
        "extra_info": "Reverse the entire string to get the original flag.",
        "difficulty": "easy",
    }


def generate_chunked_flag_challenge():
    flag = generate_flag()

    chunks = []
    i = 0
    while i < len(flag):
        chunk_len = random.choice([2, 3])
        chunks.append(flag[i:i + chunk_len])
        i += chunk_len

    ciphertext = " ".join(chunks)

    return {
        "flag": flag,
        "ciphertext": ciphertext,
        "cipher": "ChunkedFlag",
        "hint": "The flag is split into parts of 2–3 letters with spaces.",
        "necessary_info": "The flag format is flag{...}",
        "extra_info": "Remove the spaces to recover the original flag.",
        "difficulty": "easy",
    }


def generate_classical_cipher_challenge(variation):
    if variation == "caesar":
        return generate_caesar_challenge()
    elif variation == "vigenere":
        return generate_vigenere_challenge()
    elif variation == "rail_fence":
        return generate_rail_fence_challenge()
    elif variation == "playfair":
        return generate_playfair_challenge()
    elif variation == "hill":
        return generate_hill_challenge()
    elif variation == "substitution":
        return generate_substitution_challenge()
    elif variation == "transposition":
        return generate_transposition_challenge()
    elif variation == "autokey":
        return generate_autokey_challenge()
    elif variation == "base64_layered":
        return generate_base64_layered_challenge()
    elif variation == "morse_code":
        return generate_morse_code_challenge()
    elif variation == "fibonacci_encoding":
        return generate_fibonacci_encoding_challenge()
    elif variation == "XOR":
        return generate_xor_challenge()
    elif variation == "Base64":
        return generate_base64_decode_challenge()
    elif variation == "Base64_layered":
        return generate_base64_layered_challenge()
    elif variation == "Base85":
        return generate_base85_decode_challenge()
    elif variation == "Base85_layered":
        return generate_base85_layered_challenge()
    elif variation == "substitution_direct":
        return generate_substitution_direct_challenge()
    elif variation == "atbash":
        return generate_atbash_challenge()
    elif variation == "hex":
        return generate_hex_challenge()
    elif variation == "ascii_shift":
        return generate_ascii_shift_challenge()
    elif variation == "split_flag":
        return generate_split_number_challenge()
    elif variation == "reversed_flag":
        return generate_reversal_challenge()
    elif variation == "chunked_flag":
        return generate_chunked_flag_challenge()
    else:
        raise ValueError(f"Unknown classical cipher variation: {variation}")

# Example Usage
if __name__ == "__main__":
    # Try each variation for testing
    for var in [
        "caesar", "vigenere", "rail_fence", 
        "playfair", "hill", "substitution", 
        "transposition", "autokey", "split_flag", "reversed_flag", "chunked_flag"
    ]:
        challenge = generate_classical_cipher_challenge(var)
        print(f"--- {var.upper()} ---")
        for k, v in challenge.items():
            print(f"{k}: {v}")
        print()
