#!/usr/bin/env bash

cd ./structural_losses/

# compile nndistance op
sh tf_nndistance_compile.sh

# compile approxmatch op
sh tf_approxmatch_compile.sh
