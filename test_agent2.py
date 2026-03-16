
import sys
from pathlib import Path

# Add the current directory to sys.path
sys.path.append(str(Path.cwd()))

from agent2_policy_checker import _check_readiness

def test():
    # Test case 1: Missing all
    ready, missing = _check_readiness({})
    print(f"Test 1 - Ready: {ready}, Missing count: {len(missing)}")
    
    # Test case 2: Presence of all
    fields = {
        "insurer_name": "BCBS",
        "policy_number": "123",
        "member_id": "M001",
        "patient_name": "John Doe"
    }
    ready, missing = _check_readiness(fields)
    print(f"Test 2 - Ready: {ready}, Missing count: {len(missing)}")

    # Test case 3: Missing policy number
    fields = {
        "insurer_name": "BCBS",
        "member_id": "M001",
        "patient_name": "John Doe"
    }
    ready, missing = _check_readiness(fields)
    print(f"Test 3 - Ready: {ready}, Missing count: {len(missing)}")
    for m in missing:
        print(f"  Missing: {m['info_needed']} (Doc: {m['document_type']})")

if __name__ == "__main__":
    test()
