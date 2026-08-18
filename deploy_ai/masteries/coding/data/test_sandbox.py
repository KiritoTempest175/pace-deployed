import subprocess
import sys

# 1. Let's create a sample code string with a DELIBERATE bug (2 + 2 = 5!)
buggy_code = """
def add_numbers(a, b):
    return a + b + 1  # BUG: Flipped logic / off-by-one error!
"""

# 2. Let's attach an objective unit test assertion
unit_test = """
assert add_numbers(2, 2) == 4, "Math failed: 2 + 2 should be 4!"
print("ALL TESTS PASSED!")
"""

# We combine them into a single executable payload
full_payload = buggy_code + "\n" + unit_test

print("Launching isolated child subprocess to verify code...")

try:
    # 3. We spawn a child python process to execute the payload with a 2-second timeout!
    result = subprocess.run(
        [sys.executable, "-c", full_payload],
        capture_output=True,  # Capture what prints to screen
        text=True,  # Decode output as strings, not raw bytes
        timeout=2.0,  # Kill process if it loops for more than 2 seconds
    )

    # 4. Evaluate the Exit Code (0 = Success, anything else = Crash/Test Failure)
    if result.returncode == 0:
        print("SANDBOX RESULT: CLEAN (All unit tests passed!)")
        print("Output:", result.stdout.strip())
    else:
        print("SANDBOX RESULT: BUG DETECTED (Unit test failed!)")
        print("Error Traceback:\n", result.stderr.strip())

except subprocess.TimeoutExpired:
    print("SANDBOX RESULT: INFINITE LOOP TERMINATED (Exceeded 2.0s limit!)")