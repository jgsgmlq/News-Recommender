"""
Fix TensorBoard Protobuf Issue
Quick script to fix the common protobuf version conflict
"""
import subprocess
import sys


def fix_protobuf():
    """Fix protobuf version issue"""
    print("=" * 70)
    print("  Fixing TensorBoard Protobuf Issue")
    print("=" * 70)
    print()
    print("This will downgrade protobuf to 3.20.3 (compatible with TensorBoard)")
    print()

    response = input("Proceed? (Y/n): ").strip().lower()
    if response == 'n':
        print("Skipped.")
        return False

    print("\nDowngrading protobuf...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "protobuf==3.20.3"],
            check=True
        )
        print("\n✓ Protobuf downgraded successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease try manually:")
        print("  pip install protobuf==3.20.3")
        return False


def verify_fix():
    """Verify the fix works"""
    print("\nVerifying fix...")
    try:
        from torch.utils.tensorboard import SummaryWriter
        print("✓ TensorBoard can be imported successfully!")
        return True
    except Exception as e:
        print(f"✗ Still has issues: {e}")
        print("\nAlternative solution:")
        print("  The training script has been updated to work WITHOUT TensorBoard")
        print("  You can run training directly, it will just skip TensorBoard logging")
        return False


def main():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║              TensorBoard Protobuf Fix Tool                    ║
╚═══════════════════════════════════════════════════════════════╝

The error you encountered is a common protobuf version conflict.

Solution 1: Downgrade protobuf (recommended)
Solution 2: Training script now works WITHOUT TensorBoard
    """)

    # Try to fix
    if fix_protobuf():
        verify_fix()
        print("\n" + "=" * 70)
        print("  Fix Applied!")
        print("=" * 70)
        print("\nYou can now run:")
        print("  python quick_test.py")
    else:
        print("\n" + "=" * 70)
        print("  No worries!")
        print("=" * 70)
        print("\nThe training script has been updated to work WITHOUT TensorBoard.")
        print("You can still run training, it will just skip TensorBoard logging.")
        print("\nRun:")
        print("  python quick_test.py")

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
