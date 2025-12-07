"""
Migration script to update existing violation and evaluation IDs to 5-digit padded format.

This script migrates:
- Violation IDs: V1, V01, V100, V146 -> V00001, V00001, V00100, V00146
- Evaluation IDs: EV1, EV01, EV100 -> EV00001, EV00001, EV00100
- Clothing Detection IDs: CD1, CD01, CD100 -> CD00001, CD00001, CD00100

Run this script once to update all existing records in the database.
"""

import sys
import os

# Add parent directory to path to import db module
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from py.db import (
    migrate_violation_ids_to_padded_format,
    migrate_evaluation_ids_to_padded_format,
    violations,
    evaluations,
)


def main():
    """Run the migration for both violations and evaluations."""
    print("=" * 60)
    print("ID Migration Script - 5-Digit Padded Format")
    print("=" * 60)
    print()
    
    # Count existing records
    violation_count = violations.count_documents({})
    evaluation_count = evaluations.count_documents({})
    
    print(f"Found {violation_count} violations in database")
    print(f"Found {evaluation_count} evaluations in database")
    print()
    
    # Ask for confirmation
    response = input("Do you want to proceed with the migration? (yes/no): ").strip().lower()
    if response not in ["yes", "y"]:
        print("Migration cancelled.")
        return
    
    print()
    print("Starting migration...")
    print("-" * 60)
    
    # Migrate violations
    print("Migrating violation IDs...")
    vio_updated = migrate_violation_ids_to_padded_format()
    print(f"✓ Updated {vio_updated} violation IDs")
    
    # Migrate evaluations
    print("Migrating evaluation IDs...")
    eval_updated = migrate_evaluation_ids_to_padded_format()
    print(f"✓ Updated {eval_updated} evaluation IDs")
    
    print("-" * 60)
    print()
    print("Migration completed successfully!")
    print(f"  - Violations updated: {vio_updated}")
    print(f"  - Evaluations updated: {eval_updated}")
    print()
    print("All new IDs will now use 5-digit padded format (V00001, EV00001, CD00001)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMigration interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during migration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

