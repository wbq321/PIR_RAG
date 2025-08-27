#!/usr/bin/env python3
"""
Detailed trace of AES-based PIR communication in Graph-PIR Phase 1

This shows exactly what data flows between client and server in each round
of the graph traversal using AES-based PianoPIR.
"""

def trace_aes_pir_communication():
    print("🔐 Graph-PIR Phase 1: AES-based PIR Communication Trace")
    print("=" * 70)
    
    print("\n📋 Setup Phase:")
    print("1. Client generates AES master key: K = random_64_bits")
    print("2. Server stores database as uint64 arrays with embeddings + neighbors")
    print("3. Client derives AES-128 key from master key for encryption")
    
    print("\n🔄 Graph Traversal Round Example:")
    print("Scenario: Client wants to query node ID = 42 to get its embedding + neighbors")
    
    print("\n" + "="*50)
    print("CLIENT → SERVER (PIR Query)")
    print("="*50)
    
    print("\n1️⃣ Client Creates PIR Query:")
    print("   Target node: 42")
    print("   Database chunks: [chunk_0, chunk_1, chunk_2, ...]")
    print("   Chunk containing node 42: chunk_1 (example)")
    
    print("\n   🎯 Real vs Dummy Offsets:")
    print("   - chunk_0: offset = AES_PRF(query_num*chunks + 0) % chunk_size  [DUMMY]")
    print("   - chunk_1: offset = 42 - chunk_1_start                          [REAL TARGET]")
    print("   - chunk_2: offset = AES_PRF(query_num*chunks + 2) % chunk_size  [DUMMY]")
    print("   ...")
    
    print("\n   📦 Query Structure:")
    query_example = {
        "timestamp": "1693123456789012",
        "target_index": "42", 
        "offsets": "[7, 15, 23, 8, 31, ...]",  # One real, others dummy
        "aes_encrypted": "True",
        "nonce": "16_random_bytes",
        "total_size": "2048 bytes"
    }
    
    for key, value in query_example.items():
        print(f"   {key}: {value}")
    
    print("\n   🔒 AES Encryption Process:")
    print("   plaintext = pack(timestamp, index=42, offsets)")
    print("   padded = pad(plaintext, AES.block_size)")
    print("   cipher = AES.new(key, AES.MODE_CTR)")
    print("   encrypted = cipher.encrypt(padded)")
    print("   final_query = nonce + encrypted + random_padding → 2048 bytes")
    
    print("\n" + "="*50)
    print("SERVER PROCESSING")
    print("="*50)
    
    print("\n2️⃣ Server Receives Query:")
    print("   Server does NOT know which offset is real!")
    print("   Server processes ALL offsets equally:")
    
    print("\n   📊 XOR Operation:")
    print("   result = 0")
    print("   for each chunk_i, offset_i in query:")
    print("       idx = chunk_i_start + offset_i")
    print("       result ^= database[idx]  # XOR database entry")
    
    print("\n   Example XOR computation:")
    print("   result = db[7] ⊕ db[chunk_1_start + 15] ⊕ db[chunk_2_start + 23] ⊕ ...")
    print("                     └─ This contains our target node 42!")
    
    print("\n   💡 Why XOR works:")
    print("   - Real entry: contains actual (embedding + neighbors) for node 42")
    print("   - Dummy entries: pseudo-random based on AES PRF")
    print("   - XOR cancels out structured patterns, hides the real target")
    
    print("\n" + "="*50)
    print("SERVER → CLIENT (PIR Response)")
    print("="*50)
    
    print("\n3️⃣ Server Response:")
    print("   Raw result: [uint64_array] # XOR of all queried entries")
    print("   Size: embedding_dim*4 + 16*4 bytes (e.g., 768*4 + 64 = 3136 bytes)")
    
    response_example = {
        "xor_result": "[uint64_0, uint64_1, ..., uint64_N]",
        "embedding_part": "First 768*4 bytes (3072 bytes)",
        "neighbors_part": "Next 16*4 bytes (64 bytes)", 
        "encryption": "XOR-based obfuscation",
        "total_size": "~3136 bytes"
    }
    
    for key, value in response_example.items():
        print(f"   {key}: {value}")
    
    print("\n" + "="*50)
    print("CLIENT DECRYPTION")
    print("="*50)
    
    print("\n4️⃣ Client Decrypts Response:")
    print("   Client knows the dummy offsets it generated with AES PRF")
    print("   Client can reconstruct dummy database entries:")
    
    print("\n   🧮 Reconstruction Process:")
    print("   dummy_xor = 0")
    print("   for each dummy_offset in dummy_offsets:")
    print("       dummy_xor ^= reconstruct_dummy_entry(dummy_offset)")
    
    print("\n   🎯 Extract Real Data:")
    print("   real_data = server_response ⊕ dummy_xor")
    print("   embedding = real_data[0:3072]  # First part")
    print("   neighbors = real_data[3072:3136]  # Second part")
    
    print("\n   📊 Final Result:")
    print("   node_42_embedding = parse_float32_array(embedding)")
    print("   node_42_neighbors = parse_int32_array(neighbors)")
    
    print("\n🔒 PRIVACY GUARANTEES:")
    print("-" * 30)
    print("✅ Server cannot tell which node was queried")
    print("✅ All offsets look equally random to server") 
    print("✅ XOR hides access pattern in database")
    print("✅ Client uses AES PRF for computational security")
    print("✅ Communication size independent of database size")
    
    print("\n📈 COMMUNICATION ANALYSIS:")
    print("-" * 30)
    print("Upload (Client → Server):")
    print("  - PIR Query: ~2048 bytes (fixed size)")
    print("  - Contains: encrypted offsets + protocol overhead")
    
    print("\nDownload (Server → Client):")
    print("  - PIR Response: ~3136 bytes")
    print("  - Contains: XOR of (embedding + neighbors)")
    print("  - Size = 768*4 + 16*4 = 3072 + 64 bytes")
    
    print("\nPer PIR Query Total: ~5184 bytes (~5KB)")
    print("Multiple queries during graph traversal multiply this cost")
    
    print("\n🎯 This is why Graph-PIR Phase 1 can have significant communication overhead!")

if __name__ == "__main__":
    trace_aes_pir_communication()
