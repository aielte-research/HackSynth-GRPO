import random
import string

def random_string(length=8):
    random_chars = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    return random_chars

def generate_flag(length=8, numeric=True):
    """
    Generates a flag string in the format flag{random_chars}.
    By default, uses 8 random alphanumeric characters inside the braces.
    """
    character_pool = string.ascii_lowercase
    
    if numeric:
        character_pool += string.digits

    random_chars = ''.join(random.choices(character_pool, k=length))
    return f"flag{{{random_chars}}}"

def pkcs7_pad(data, block_size=16):
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len]) * pad_len

