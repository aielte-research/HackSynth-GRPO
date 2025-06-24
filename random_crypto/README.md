# 🧪 Random-Crypto Benchmark

The Random-Crypto Benchmark generates cryptographic CTF challenges tailored for evaluating and training large language models in reinforcement learning settings.

It includes two pre-generated sets of problems:

* ✅ 50 Human-verified challenges for evaluation [(link)](https://github.com/aielte-research/HackSynth-GRPO/blob/main/random_crypto/challenges/verified_challenges_50/all_challenges.csv)
* ⚙️ 5000 Non-Verified Challenges for training [(link)](https://github.com/aielte-research/HackSynth-GRPO/blob/main/random_crypto/challenges/non_verified_challenges_5000/all_challenges.csv)

> 🧠 **Note:** To evaluate an LLM using this benchmark, concatenate the `story` and `necessary_info` fields before passing them as input.

---

## 📊 CSV Format

Each row in the challenge CSV files contains a single cryptographic task with the following columns:

- `story`: The main challenge description presented to the model.
- `necessary_info`: Key information required to solve the task.
- `hint`: (Optional) A short hint to help solve the challenge.
- `flag`: The expected correct solution (flag) in the format `flag{...}`.
- `cipher`: The type of cipher or cryptographic scheme used (e.g., Caesar, Vigenère).
- `extra_info`: Additional metadata, e.g., internal generation parameters or the used key.
- `difficulty`: One of `easy`, `medium`, or `hard`.

---

## ⚙️ Generating New Challenges

If you want to generate new or additional procedural challenges, use the `main.py` script.

Make sure to set your **OpenAI API key** in a `.env` file at the root of this folder:

```env
OPENAI_API_KEY=your-key-here
```

### Arguments

| Argument           | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `--variants`       | Number of variants per challenge subtype. <br>**Default:** `1`              |
| `--output_folder`  | Folder where generated challenges and metadata will be saved. <br>**Default:** `"generated_challenges_llm"` |


### Example Usage
This code generates 50 challenges, one from each type.

```bash
python main.py --variants 1 --output_folder my_generated_challenges
```

### Challenge Taxonomy

The following table summarizes the types of cryptographic challenges currently supported in the benchmark:

| **Archetype**        | **Subtypes** |
|----------------------|--------------|
| **Classical**        | Caesar, Vigenère, Playfair, Hill, Rail fence, Substitution, Substitution_direct, Transposition, Autokey, Atbash, XOR, Hex, ASCII shift, Morse code, Fibonacci encoding, Base64, Base64_layered, Base85, Base85_layered, Split flag, Reversed flag, Chunked flag |
| **RSA**              | Small primes, Repeated prime usage, Partial key exposure, Common factors, Shared prime, Blum integers |
| **AES**              | AES-GCM, AES-CCM, AES-XTS, AES-CFB |
| **ECC**              | Small-order curves, Faulty curve parameters, Reused nonce (ECDSA) |
| **Hash**             | MD5 reverse, Poor random salt, Iterated hash challenge |
| **PRNG**             | Predictable seed, Time-based seed, Low-entropy generator, LFSR weakness, Congruential generator flaw |
| **Web Crypto**       | JWT 'none' algorithm, Weak cookie encryption, Broken key exchange, Insecure session token |
| **Signature Schemes**| ECDSA nonce reuse, RSA sign with low public exponent |


---

## Contributors
- Lajos Muzsai (muzsailajos@protonmail.com)
- David Imolai (david@imol.ai)
- András Lukács (andras.lukacs@ttk.elte.hu)

---
## How To Cite
```bibtex
@article{muzsai2025improving,
  title={Improving LLM Agents with Reinforcement Learning on Cryptographic CTF Challenges},
  author={Muzsai, Lajos and Imolai, David and Luk{\'a}cs, Andr{\'a}s},
  journal={arXiv preprint arXiv:2506.02048},
  year={2025}
}
```
