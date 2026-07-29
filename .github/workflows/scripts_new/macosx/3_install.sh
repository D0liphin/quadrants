#!/bin/bash

set -ex

# Diagnostics branch: install a prebuilt quadrants wheel from PyPI instead of
# building from source (saves the ~15-18 min Mac build). QD_VERSION pins a specific
# release so we can bracket the CI-slowdown regression across versions; blank
# installs the latest available (including pre-releases).
if [ -n "${QD_VERSION}" ]; then
  pip install --prefer-binary "quadrants==${QD_VERSION}"
else
  pip install --prefer-binary --pre --upgrade quadrants
fi

python -c "import quadrants as qd; qd.init(arch=qd.cpu)"
python -c "import quadrants as qd; qd.init(arch=qd.metal)"
