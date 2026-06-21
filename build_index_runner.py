"""直接运行索引构建（带日志）。"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.cli.build_index import main

if __name__ == "__main__":
    main()
