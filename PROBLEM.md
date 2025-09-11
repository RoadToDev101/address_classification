**Address Classification Problem**

Classify an input address string into province, district, and ward, despite inconsistencies, spelling errors, and missing data. Use three Python lists (`provinces`, `districts`, `wards`) without hierarchical structure (no mapping between province, district, ward).

**Input**: Address string (e.g., "Xã Thịnh Sơn H., Đô dương T. Nghệ An")  
**Output**: Dictionary with keys `province`, `district`, `ward` (e.g., `{"province": "Nghệ An", "district": "Đô Lương", "ward": "Thịnh Sơn"}`). Set missing components to `null`.

**Constraints**:  
- Max time per request: ≤0.1s  
- Avg time per request: ≤0.01s  
- No machine learning; use algorithmic approach only  
- Use Dynamic Programming (recommended by instructor)  
- Handle spelling errors and inconsistencies  
- Missing data in input (e.g., no province) should not prevent recognizing district/ward; output `null` for missing components  

**Data**:  
- `provinces`: List of province names (e.g., `['Ha Noi', 'Nghe An', 'Ho Chi Minh', 'Vinh Long', ...]`)  
- `districts`: List of district names  
- `wards`: List of ward names  
- No hierarchical relationships between lists  

**Example**:  
```json
{
  "input": "Xã Thịnh Sơn H., Đô dương T. Nghệ An",
  "expected_output": {
    "province": "Nghệ An",
    "district": "Đô Lương",
    "ward": "Thịnh Sơn"
  }
}
```

**Tips**: Trie-based approach (inspired by autocomplete or spell correction) may be viable for handling spelling errors and efficient matching.