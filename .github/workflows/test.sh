#!/usr/bin/env bash
  
set -e

cd ./pyscf
pytest -c pytest.ini
