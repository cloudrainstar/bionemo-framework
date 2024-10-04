#!/bin/bash

for sub in `ls -d ./3rdparty/*` `ls -d./sub-packages/*`; do
    uv pip install --no-deps --no-build-isolation --editable $sub
done
