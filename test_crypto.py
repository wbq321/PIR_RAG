#!/usr/bin/env python3
"""
Test the SimpleLinearHomomorphicScheme to verify the scalar multiplication is correct.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from tiptoe.crypto_fixed import SimpleLinearHomomorphicScheme

def test_scalar_multiplication():
    """Test that scalar multiplication works correctly."""
    print("Testing SimpleLinearHomomorphicScheme scalar multiplication...")

    crypto = SimpleLinearHomomorphicScheme()

    # Test cases
    test_cases = [
        (5, 3),    # 5 * 3 = 15
        (0, 7),    # 0 * 7 = 0
        (10, 0),   # 10 * 0 = 0
        (1, 100),  # 1 * 100 = 100
        (25, 4),   # 25 * 4 = 100
    ]

    print(f"Secret key: {crypto.secret_key}")
    print(f"Modulus: {crypto.modulus}")
    print()

    all_passed = True

    for plaintext, scalar in test_cases:
        print(f"Testing: {scalar} * Enc({plaintext})")

        # 1. Encrypt the plaintext
        encrypted = crypto.encrypt(plaintext)
        print(f"  Enc({plaintext}) = {encrypted['value']}")

        # 2. Verify encryption/decryption works
        decrypted_original = crypto.decrypt(encrypted)
        print(f"  Dec(Enc({plaintext})) = {decrypted_original}")

        if decrypted_original != plaintext:
            print(f"  ❌ FAIL: Encryption/decryption doesn't work!")
            all_passed = False
            continue

        # 3. Perform scalar multiplication
        multiplied = crypto.scalar_multiply(encrypted, scalar)
        print(f"  {scalar} * Enc({plaintext}) = {multiplied['value']}")

        # 4. Decrypt the result
        decrypted_result = crypto.decrypt(multiplied)
        expected = (scalar * plaintext) % crypto.modulus

        print(f"  Dec({scalar} * Enc({plaintext})) = {decrypted_result}")
        print(f"  Expected: {scalar} * {plaintext} = {expected}")

        if decrypted_result == expected:
            print(f"  ✅ PASS")
        else:
            print(f"  ❌ FAIL: Got {decrypted_result}, expected {expected}")
            all_passed = False

        print()

    if all_passed:
        print("🎉 All tests passed! Scalar multiplication is working correctly.")
    else:
        print("💥 Some tests failed! There's a bug in the scalar multiplication.")

    return all_passed

def test_homomorphic_properties():
    """Test homomorphic addition as well."""
    print("Testing homomorphic addition...")

    crypto = SimpleLinearHomomorphicScheme()

    # Test: Enc(a) + Enc(b) = Enc(a + b)
    a, b = 15, 25

    enc_a = crypto.encrypt(a)
    enc_b = crypto.encrypt(b)

    # Homomorphic addition
    enc_sum = crypto.add_encrypted(enc_a, enc_b)
    decrypted_sum = crypto.decrypt(enc_sum)
    expected_sum = (a + b) % crypto.modulus

    print(f"Enc({a}) + Enc({b}) = Enc({expected_sum})")
    print(f"Dec(Enc({a}) + Enc({b})) = {decrypted_sum}")
    print(f"Expected: {expected_sum}")

    if decrypted_sum == expected_sum:
        print("✅ Homomorphic addition works correctly!")
        return True
    else:
        print("❌ Homomorphic addition is broken!")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("SIMPLE LINEAR HOMOMORPHIC SCHEME VERIFICATION")
    print("=" * 60)

    scalar_test_passed = test_scalar_multiplication()
    print()
    addition_test_passed = test_homomorphic_properties()

    print()
    if scalar_test_passed and addition_test_passed:
        print("🎉 All cryptographic tests passed!")
    else:
        print("💥 Cryptographic scheme has bugs!")
