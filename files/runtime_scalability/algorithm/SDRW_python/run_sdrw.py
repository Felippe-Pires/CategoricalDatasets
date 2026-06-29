#!/usr/bin/env python3
"""
Convenience launcher for the SDRW Python port.

Usage:
  python run_sdrw.py [data_path_or_dir] [feature_selection=true|false] 

Environment:
  SDRW_DATA_DIR    — default dataset directory or file
  SDRW_RESULT_DIR  — directory for CSV outputs
"""

from sdrw.dsvl4od_utils import main

if __name__ == "__main__":
    main()
